import requests
import re
import os
import pickle
import pandas as pd
import openai
import json
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from flask import Flask, request, jsonify

script_directory = os.path.dirname(__file__)
column_transformer = os.path.join(script_directory, 'models', 'column_transformer.sav')
model_file = os.path.join(script_directory, 'models', 'model.sav')
system_prompt_file = os.path.join(script_directory, 'prompts', 'system-prompt.txt')

app = Flask(__name__)

with open('config.json') as config_file:
  config = json.load(config_file)

def get_embeddings(text: str):
    url = "https://api.openai.com/v1/embeddings"
    api_key = config["embedding_key"]

    # Define the request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Define the request payload as a dictionary
    payload = {
        "input": text,
        "model": "text-embedding-ada-002"
    }
    response = requests.post(url, json=payload, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        embedding_data = response.json()
        embedding = embedding_data["data"][0]["embedding"]
    else:
        print("Error:", response.status_code, response.text)
    
    return embedding

def single_vector_search(query: str):
    search_client = SearchClient(config["cognitive_search_service_endpoint"], config["cognitive_search_index_name"], AzureKeyCredential(config["search_client_key"]))
    vector = Vector(value=get_embeddings(query), k=100, fields="embedding")

    results = search_client.search(
        search_text="",
        vectors=[vector],
        select=["video_title", "video_id", "thumbnail_description", "thumbnail_text", "video_transcript", "video_rating"],
    )

    results = list(results)
    results_dict = {}

    sorted_results = sorted(results, key=lambda x: x["video_rating"], reverse=True)
    top_10_highest_rated = sorted_results[:10]
    
    results_dict = {
      "closest_results": results,
      "highest_rated_results": top_10_highest_rated
    }

    return results_dict

def generate_concatenated_details(obj):
  return (
      f"[Title]: {obj['video_title']}\n"
      f"[Thumbnail Description]: {obj['thumbnail_description']}\n"
      f"[Thumbnail Text]: {obj['thumbnail_text']}\n"
      f"[Transcript]: {obj['video_transcript']}]\n"
      f"[Video Rating]: {obj['video_rating']}\n"
  )

def parse_youtube_duration(duration_str):
    # Parse YouTube duration format (e.g., 'PT2H30M15S' for 2 hours, 30 minutes, and 15 seconds)
    regex = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(regex, duration_str)

    if match:
        hours, minutes, seconds = map(int, [group or 0 for group in match.groups()])
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds

def get_video_data(video_id): 
    print(f'Getting data for video {video_id}')
    youtube = build('youtube', 'v3', developerKey=config["youtube_api_key"])
    video_response = youtube.videos().list(
        id=video_id,
        part='snippet,contentDetails'
    ).execute()

    if video_response['items']:
        video_info = video_response['items'][0]
        return video_info

    return None

def get_thumbnail_data(video):
    print(f'Getting thumbnail data for video {video["video_id"]}')
    headers = {
        "Ocp-Apim-Subscription-Key": config["image_analysis_api_key"],
        "Content-Type": "application/json"
    }
    params = {
        "features": "caption,read,tags",
        "model-version": "latest",
        "language": "en",
        "api-version": "2023-02-01-preview"
    }
    data = {
      "url": video['video_thumbnail']
    }
    response = requests.post(config["image_analysis_endpoint"], headers=headers, params=params, json=data)

    video["thumbnail_description"] = response.json()["captionResult"]["text"]
    video["thumbnail_text"] = response.json()["readResult"]["content"]
    
    thumbnail_tags = []

    for tag in response.json()["tagsResult"]["values"]:
      if tag["confidence"] > 0.9:
        thumbnail_tags.append(tag["name"])

    video["thumbnail_tags"] = " ,".join(thumbnail_tags)

    return video

def get_video_transcript(video):
    print(f'Getting video transcript for video {video["video_id"]}')
    generatedTranscript = YouTubeTranscriptApi.get_transcript(video['video_id'])
    sorted_transcript = sorted(generatedTranscript, key=lambda x: x['start'])
    final_transcript = ""

    for line in sorted_transcript:
      final_transcript += line['text'].replace('\n', ' ') + " "

    video["video_transcript"] = final_transcript

    return video

def generate_rating(video):
    print(f'Generating rating for video {video["video_id"]}')
    with open(column_transformer, "rb") as ct_file:
      ct = pickle.load(ct_file)
    
    df1 = pd.DataFrame([video])

    X = df1[["video_title", "thumbnail_description", "thumbnail_tags", "video_duration", "video_transcript"]]
    X_ct = ct.transform(X)

    model = pickle.load(open(model_file, 'rb'))

    rating = model.predict(X_ct)
    return rating

def normalise_rating(generated_rating, related_videos):
    related_ratings = []
    for video in related_videos:
      related_ratings.append(video["video_rating"])
    
    mean_ratio = np.mean(related_ratings)
    std_deviation = np.std(related_ratings)

    z_score = (generated_rating - mean_ratio) / std_deviation
    rating = (z_score * 5) + 5
    rating = max(0, min(10, rating))
    rating = round(float(rating.item()), 2)
    return rating

def get_suggestions_and_violations(user_video, related_videos):
  user_prompt = "Hello, I would like to know how I can improve this video. The details are"
  with open(system_prompt_file) as f:
    system_prompt = f.read()

  user_prompt += "{" + user_video + "}"
  for video in related_videos:
    system_prompt += "{" + generate_concatenated_details(video) + "}"
  
  openai.api_key = config["gpt_api_key"]
  openai.api_type = "azure"
  openai.api_base = "https://ausopenai.azure-api.net"
  openai.api_version = "2023-05-15"

  response = openai.ChatCompletion.create(
      engine="gpt-35-turbo-16k",
      messages=[
          {
            "role": "system", 
            "content": system_prompt
          },
          {"role": "user", "content": user_prompt}
      ]
  )

  return response['choices'][0]['message']['content']   

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/video', methods=['GET'])
def process_data():
    
        # Get input data from the query parameter 'data'
        input_video = request.args.get('video')
        print(f'Processing video {input_video}')
        # Call a function in another_file.py to process the input data
        parsed_url = urlparse(input_video)
        video_id = parse_qs(parsed_url.query).get('v')[0]
        video_rating = 0

        if video_id:
            video_data = get_video_data(video_id)
            if video_data:
                video = {
                    'video_title': video_data['snippet']['title'],
                    'video_id': video_id,
                    'video_thumbnail': video_data['snippet']['thumbnails']['high']['url'],
                    'video_duration': parse_youtube_duration(video_data['contentDetails']['duration']),
                }
                video = get_thumbnail_data(video)
                video = get_video_transcript(video)
                
                video_rating = generate_rating(video)
                video["video_rating"] = video_rating
                video["thumbnail_tags"] = " ,".join(video["thumbnail_tags"])

                concatenated_video = generate_concatenated_details(video)
                related_videos = single_vector_search(concatenated_video)
                suggestions = get_suggestions_and_violations(concatenated_video, related_videos.get("highest_rated_results"))
                normalised_rating = normalise_rating(video_rating, related_videos.get("closest_results"))
                output_data = {
                    'success': 'true',
                    'suggestions': suggestions,
                    'rating': normalised_rating
                }
            else:
                print(f'Could not find video with id {video_id}')
                output_data = {
                    'success': 'false',
                    'suggestions': '',
                    'rating': 0,
                    'error': f'Could not find video with id {video_id}'
                }
        else:
            print('Could not parse video id from provided input')
            output_data = {
                'success': 'false',
                'suggestions': '',
                'rating': 0,
                'error': 'Could not parse video id from provided input'
            }

        # Return the processed output as JSON
        return jsonify(output_data)    

if __name__ == '__main__':
    if __name__ == "__main__":
	    app.run(host="127.0.0.1", port=8000)

