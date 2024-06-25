import logging
import traceback
from flask import Flask, redirect, render_template, request, jsonify, url_for
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Set up basic logging
logging.getLogger('watchdog').setLevel(logging.WARNING)
app = Flask(__name__)

# Load the datasets
try:
    conversations_df = pd.read_csv('conversations.csv')
    users_df = pd.read_csv('users.csv')
    interactions_df = pd.read_csv('interactions.csv')
except FileNotFoundError:
    conversations_df = pd.DataFrame(columns=['id', 'topic', 'content'])
    users_df = pd.DataFrame(columns=['id', 'username', 'age', 'interests'])
    interactions_df = pd.DataFrame(columns=['user_id', 'conversation_id', 'interaction_type'])


# Idiom dataset
english_idioms = [
    {"idiom": "A piece of cake", "meaning": "Something very easy to do", "min_age": 8},
    {"idiom": "Break a leg", "meaning": "Good luck", "min_age": 10},
    {"idiom": "Hit the nail on the head", "meaning": "To describe exactly what is causing a situation or problem", "min_age": 12},
    {"idiom": "It's raining cats and dogs", "meaning": "It's raining heavily", "min_age": 7},
    {"idiom": "Jump on the bandwagon", "meaning": "Join a popular trend or activity", "min_age": 13},
    {"idiom": "Kill two birds with one stone", "meaning": "Achieve two things with a single action", "min_age": 10},
    {"idiom": "Let the cat out of the bag", "meaning": "Reveal a secret accidentally", "min_age": 9},
    {"idiom": "Once in a blue moon", "meaning": "Very rarely", "min_age": 8},
    {"idiom": "Speak of the devil", "meaning": "The person we were just talking about has appeared", "min_age": 11},
    {"idiom": "The ball is in your court", "meaning": "It's your turn to take action or make a decision", "min_age": 12},
    {"idiom": "Barking up the wrong tree", "meaning": "Looking for a solution in the wrong place", "min_age": 14},
    {"idiom": "Cost an arm and a leg", "meaning": "To be very expensive", "min_age": 10},
    {"idiom": "Don't put all your eggs in one basket", "meaning": "Don't risk everything on a single venture", "min_age": 13},
    {"idiom": "Elephant in the room", "meaning": "An obvious problem that people avoid discussing", "min_age": 15},
    {"idiom": "Get your act together", "meaning": "Organize yourself and behave more effectively", "min_age": 11},
    {"idiom": "Beat around the bush", "meaning": "Avoid talking about something directly", "min_age": 12},
    {"idiom": "Bite off more than you can chew", "meaning": "Take on a task that is too big to handle", "min_age": 11},
    {"idiom": "Break the ice", "meaning": "Make people feel more comfortable in a social situation", "min_age": 10},
    {"idiom": "Cut corners", "meaning": "Do something in the easiest or cheapest way", "min_age": 13},
    {"idiom": "Get cold feet", "meaning": "Become nervous about doing something", "min_age": 12},
    {"idiom": "Hit the books", "meaning": "Study hard", "min_age": 10},
    {"idiom": "In hot water", "meaning": "In trouble", "min_age": 9},
    {"idiom": "Keep your chin up", "meaning": "Stay positive in difficult times", "min_age": 11},
    {"idiom": "Make a long story short", "meaning": "Tell something briefly", "min_age": 10},
    {"idiom": "On cloud nine", "meaning": "Extremely happy", "min_age": 9},
    {"idiom": "Pull someone's leg", "meaning": "Joke with someone", "min_age": 10},
    {"idiom": "Raining cats and dogs", "meaning": "Raining heavily", "min_age": 8},
    {"idiom": "Spill the beans", "meaning": "Reveal a secret", "min_age": 9},
    {"idiom": "Take a rain check", "meaning": "Postpone an invitation", "min_age": 13},
    {"idiom": "Under the weather", "meaning": "Feeling ill", "min_age": 10},
]

# Your existing functions go here
def content_based_recommendations(topics, n=5):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(conversations_df['content'])
    
    # Get the average vector for the selected topics
    topic_vectors = tfidf_matrix[conversations_df['topic'].isin(topics)]
    if topic_vectors.shape[0] == 0:
        return []  # Return empty list if no matching topics found
    avg_topic_vector = np.asarray(topic_vectors.mean(axis=0)).flatten()
    
    # Ensure avg_topic_vector is a 2D array
    avg_topic_vector = avg_topic_vector.reshape(1, -1)
    
    # Convert sparse matrix to dense numpy array
    tfidf_matrix_dense = np.asarray(tfidf_matrix.todense())
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(avg_topic_vector, tfidf_matrix_dense)
    
    # Get recommendations
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:n]
    conversation_indices = [i[0] for i in sim_scores]
    
    return conversations_df['id'].iloc[conversation_indices].tolist()

def collaborative_filtering_recommendations(user_id, n=5):
    # Aggregate interactions
    aggregated_interactions = interactions_df.groupby(['user_id', 'conversation_id'])['interaction_type'].agg(lambda x: x.value_counts().index[0]).reset_index()
    
    # Create user-item matrix
    user_item_matrix = aggregated_interactions.pivot(index='user_id', columns='conversation_id', values='interaction_type').fillna(0)
    
    # Convert interaction types to numeric values
    interaction_types = {'view': 1.0, 'like': 2.0, 'comment': 3.0}
    user_item_matrix = user_item_matrix.replace(interaction_types)
    
    # Ensure the user exists in the matrix
    if user_id not in user_item_matrix.index:
        return []  # Return empty list if user doesn't exist
    
    # Calculate user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    
    # Get similar users
    similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False)[1:11].index  # top 10 similar users
    
    # Get recommendations from similar users
    recommendations = []
    for similar_user in similar_users:
        user_conversations = set(aggregated_interactions[aggregated_interactions['user_id'] == similar_user]['conversation_id'])
        current_user_conversations = set(aggregated_interactions[aggregated_interactions['user_id'] == user_id]['conversation_id'])
        new_conversations = user_conversations - current_user_conversations
        recommendations.extend(list(new_conversations))
    
    # Remove duplicates and limit to n recommendations
    recommendations = list(dict.fromkeys(recommendations))[:n]
    
    return recommendations

def hybrid_recommendations(user_id, name, age, topics, occupation, n=5):
    logging.info(f"Starting hybrid_recommendations for user_id={user_id}, name={name}, age={age}, topics={topics}, occupation={occupation}, n={n}")
    
    content_based_rec = content_based_recommendations(topics, n*2)
    logging.info(f"Content-based recommendations: {content_based_rec}")
    
    collaborative_rec = collaborative_filtering_recommendations(user_id, n*2)
    logging.info(f"Collaborative filtering recommendations: {collaborative_rec}")
    
    # Combine recommendations
    hybrid_rec = list(set(content_based_rec + collaborative_rec))
    logging.info(f"Combined recommendations: {hybrid_rec}")
    
    # If hybrid_rec is empty, return collaborative recommendations
    if not hybrid_rec:
        logging.info("Hybrid recommendations empty, returning collaborative recommendations")
        return collaborative_rec[:n]
    
    # Boost recommendations related to user's occupation, selected topics, and age
    boosted_rec = []
    for rec_id in hybrid_rec:
        conversation = conversations_df[conversations_df['id'] == rec_id].iloc[0]
        score = 1
        if occupation.lower() in conversation['content'].lower() or occupation.lower() in conversation['topic'].lower():
            score += 1  # Boost for occupation
        if conversation['topic'] in topics:
            score += 1  # Boost for selected topics
        
        # Age-based boosting
        if 'age_group' in conversation:
            if age < 18 and conversation['age_group'] == 'youth':
                score += 1
            elif 18 <= age < 30 and conversation['age_group'] == 'young_adult':
                score += 1
            elif 30 <= age < 50 and conversation['age_group'] == 'adult':
                score += 1
            elif age >= 50 and conversation['age_group'] == 'senior':
                score += 1
        
        boosted_rec.append((rec_id, score))
    
    # Sort by score and return top n
    boosted_rec.sort(key=lambda x: x[1], reverse=True)
    final_recommendations = [rec[0] for rec in boosted_rec[:n]]
    logging.info(f"Final recommendations: {final_recommendations}")
    return final_recommendations

def recommend_idioms(age, n=5):
    suitable_idioms = [idiom for idiom in english_idioms if idiom['min_age'] <= age]
    recommended = sorted(suitable_idioms, key=lambda x: x['min_age'], reverse=True)[:n]
    return recommended

@app.route('/')
def home():
    topics = conversations_df['topic'].unique().tolist()
    return render_template('home.html', topics=topics)

@app.route('/user_input', methods=['GET', 'POST'])
def user_input():
    global users_df
    topics = conversations_df['topic'].unique().tolist()
    
    if request.method == 'POST':
        user_id = len(users_df) + 1
        username = request.form['username']
        age = int(request.form['age'])
        interests = request.form.getlist('interests')
        occupation = request.form['occupation']
        
        new_user = pd.DataFrame({
            'id': [user_id],
            'username': [username],
            'age': [age],
            'interests': [','.join(interests)],
            'occupation': [occupation]
        })
        
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv('users.csv', index=False)
        
        return redirect(url_for('recommend', user_id=user_id))
    
    return render_template('user_input.html', topics=topics)

@app.route('/get_recommendations', methods=['GET', 'POST'])
def get_recommendations():
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        n = request.form.get('n', 5)
        return redirect(url_for('recommend', user_id=user_id, n=n))
    return render_template('get_recommendations.html')

@app.route('/recommend')
def recommend():
    try:
        user_id = int(request.args.get('user_id', 1))
        n = int(request.args.get('n', 5))

        user = users_df[users_df['id'] == user_id].iloc[0]
        name = user['username']
        age = user['age']
        topics = user['interests'].split(',')
        occupation = user['occupation']

        logging.info(f"Received request: user_id={user_id}, name={name}, age={age}, topics={topics}, occupation={occupation}, n={n}")

        recommendations = hybrid_recommendations(user_id, name, age, topics, occupation, n)
        
        logging.info(f"Generated recommendations: {recommendations}")

        result = []
        for rec_id in recommendations:
            conversation = conversations_df[conversations_df['id'] == rec_id]
            if not conversation.empty:
                conversation = conversation.iloc[0]
                result.append({
                    "id": rec_id,
                    "topic": conversation['topic'],
                    "content": conversation['content'][:200] + '...' if len(conversation['content']) > 200 else conversation['content']
                })

        # Get idiom recommendations
        idiom_recommendations = recommend_idioms(age, n=3)

        return render_template('recommend.html', name=name, user_id=user_id, age=age, topics=', '.join(topics), occupation=occupation, recommendations=result, idiom_recommendations=idiom_recommendations)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        return f"An error occurred: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}", 400

if __name__ == '__main__':
    app.run(debug=True)