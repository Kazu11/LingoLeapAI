import logging
import traceback
from flask import Flask, redirect, render_template_string, request, jsonify, url_for
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

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

@app.route('/', methods=['GET', 'POST'])
def home():
    global users_df
    topics = conversations_df['topic'].unique().tolist()
    
    if request.method == 'POST':
        if 'username' in request.form:
            # User input form submission
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
        else:
            # Recommendation form submission
            user_id = request.form.get('user_id')
            n = request.form.get('n', 5)
            return redirect(url_for('recommend', user_id=user_id, n=n))
    
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>User Input and Recommendation System</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
            form { margin-top: 20px; }
            input, select { margin: 10px 0; padding: 5px; width: 200px; }
            input[type="submit"] { cursor: pointer; width: auto; }
        </style>
    </head>
    <body>
        <h1>User Input</h1>
        <form method="POST">
            Username: <input type="text" name="username" required><br>
            Age: <input type="number" name="age" required><br>
            Occupation: <input type="text" name="occupation" required><br>
            Interests (select at least 2):<br>
            <select name="interests" multiple required>
                {% for topic in topics %}
                    <option value="{{ topic }}">{{ topic }}</option>
                {% endfor %}
            </select><br>
            <input type="submit" value="Submit">
        </form>

        <h1>Get Recommendations</h1>
        <form method="POST">
            <label for="user_id">User ID:</label><br>
            <input type="number" id="user_id" name="user_id" required><br>
            <label for="n">Number of recommendations:</label><br>
            <input type="number" id="n" name="n" value="5"><br>
            <input type="submit" value="Get Recommendations">
        </form>
    </body>
    </html>
    """
    return render_template_string(html, topics=topics)

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

        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Recommendations</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }
                h1 { color: #333; }
                .recommendation { border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; }
                .recommendation h2 { color: #0066cc; margin-top: 0; }
                .recommendation p { margin: 5px 0; }
            </style>
        </head>
        <body>
            <h1>Recommendations for {{ name }} (User ID: {{ user_id }})</h1>
            <p>Age: {{ age }}</p>
            <p>Based on topics: {{ topics }}</p>
            <p>Occupation: {{ occupation }}</p>
            {% for item in recommendations %}
            <div class="recommendation">
                <h2>#{{ item['id'] }}: {{ item['topic'] }}</h2>
                <p>{{ item['content'] }}</p>
            </div>
            {% endfor %}
            <a href="/">Back to Home</a>
        </body>
        </html>
        """
        return render_template_string(html, name=name, user_id=user_id, age=age, topics=', '.join(topics), occupation=occupation, recommendations=result)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        return f"An error occurred: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}", 400

if __name__ == '__main__':
    app.run(debug=True)