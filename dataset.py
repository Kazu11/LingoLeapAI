import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_conversations(n=100):
    topics = ['technology', 'sports', 'politics', 'entertainment', 'science', 'food', 'travel', 'health']
    
    conversation_starters = {
    'technology': [
        "What do you think about the latest iPhone?",
        "How is AI changing our daily lives?",
        "What's your favorite coding language and why?",
        "How do you think 5G will impact internet usage?",
        "What are your thoughts on the future of autonomous vehicles?",
        "How do you protect your online privacy?",
        "What's your opinion on the rise of smart home devices?",
        "How do you see blockchain technology evolving in the next decade?",
        "What's your take on the ethical implications of gene editing?",
        "How do you think virtual reality will change education?",
        "What's your favorite tech gadget and why?",
        "How do you stay updated with the latest tech trends?",
        "What do you think about the concept of a universal basic income in the age of automation?"
    ],
    'sports': [
        "Who do you think will win the World Cup this year?",
        "What's your opinion on athlete salaries?",
        "How do you stay motivated to exercise regularly?",
        "What's your favorite sport to watch and why?",
        "How do you think technology is changing sports?",
        "What's your opinion on the use of instant replay in sports?",
        "Who do you consider the greatest athlete of all time and why?",
        "How do you think sports can promote social change?",
        "What's your take on the issue of doping in professional sports?",
        "How do you think the pandemic has affected the world of sports?",
        "What's your favorite sports memory?",
        "How do you think e-sports compare to traditional sports?",
        "What's your opinion on the pay gap between male and female athletes?"
    ],
    'politics': [
        "What are your thoughts on the current economic policies?",
        "How can we improve voter turnout?",
        "What role should social media play in politics?",
        "How do you think climate change should be addressed politically?",
        "What's your opinion on the current state of healthcare policy?",
        "How do you think education policy can be improved?",
        "What are your thoughts on immigration reform?",
        "How do you think political polarization can be reduced?",
        "What's your take on campaign finance reform?",
        "How do you think foreign policy should be approached in the current global climate?",
        "What are your thoughts on the role of lobbying in politics?",
        "How do you think we can increase political engagement among young people?",
        "What's your opinion on term limits for elected officials?"
    ],
    'entertainment': [
        "What's the best movie you've seen recently?",
        "How has streaming changed the TV industry?",
        "Who's your favorite musician and why?",
        "What do you think about the current trend of rebooting old TV shows and movies?",
        "How do you think social media has impacted celebrity culture?",
        "What's your opinion on the rise of user-generated content platforms like TikTok?",
        "Who do you think is the most influential figure in entertainment right now?",
        "How do you think virtual concerts compare to live performances?",
        "What's your take on the increasing diversity in Hollywood productions?",
        "How do you think the pandemic has affected the film industry?",
        "What's your favorite book-to-movie adaptation and why?",
        "How do you think video games are influencing other forms of entertainment?",
        "What's your opinion on the current state of reality TV?"
    ],
    'science': [
        "What recent scientific discovery excites you the most?",
        "How can we make STEM fields more inclusive?",
        "What's your take on space exploration?",
        "How do you think AI will impact scientific research?",
        "What's your opinion on the current state of climate science?",
        "How do you think we can improve science education?",
        "What's your take on the ethical implications of genetic engineering?",
        "How do you think quantum computing will change scientific research?",
        "What's your opinion on the role of citizen science in research?",
        "How do you think we can address the replication crisis in science?",
        "What's your take on the potential of renewable energy technologies?",
        "How do you think neuroscience is changing our understanding of consciousness?",
        "What's your opinion on the current state of vaccine development?"
    ],
    'food': [
        "What's your favorite cuisine and why?",
        "How has the pandemic affected your eating habits?",
        "What's your go-to recipe for impressing guests?",
        "How do you think climate change will impact global food production?",
        "What's your opinion on plant-based meat alternatives?",
        "How do you think technology is changing the restaurant industry?",
        "What's your favorite food trend of the past year?",
        "How do you think we can address global food waste?",
        "What's your take on the slow food movement?",
        "How do you think social media has influenced food culture?",
        "What's your opinion on genetically modified foods?",
        "How do you think we can promote healthier eating habits?",
        "What's your favorite cooking show and why?"
    ],
    'travel': [
        "What's the most memorable place you've visited?",
        "How do you think travel will change post-pandemic?",
        "What's your dream vacation destination?",
        "How do you think sustainable tourism can be promoted?",
        "What's your favorite travel experience and why?",
        "How do you think technology is changing the way we travel?",
        "What's your opinion on the impact of mass tourism on local cultures?",
        "How do you prepare for a trip to a new destination?",
        "What's your take on the rise of digital nomads?",
        "How do you think we can make travel more accessible to everyone?",
        "What's your favorite travel hack or tip?",
        "How do you think climate change will impact travel in the future?",
        "What's your opinion on voluntourism?"
    ],
    'health': [
        "What's your best tip for maintaining mental health?",
        "How do you balance work and personal wellness?",
        "What's your opinion on alternative medicine?",
        "How do you think technology is changing healthcare?",
        "What's your take on the importance of sleep for overall health?",
        "How do you think we can improve global access to healthcare?",
        "What's your opinion on the role of diet in preventing chronic diseases?",
        "How do you stay motivated to maintain a healthy lifestyle?",
        "What's your take on the impact of stress on physical health?",
        "How do you think we can improve mental health awareness and support?",
        "What's your opinion on the role of exercise in maintaining good health?",
        "How do you think we can address the global obesity epidemic?",
        "What's your take on the future of personalized medicine?"
    ]
}

    conversations = []
    for i in range(n):
        topic = np.random.choice(topics)
        content = random.choice(conversation_starters[topic])
        date = datetime.now() - timedelta(days=np.random.randint(0, 365))
        
        # Generate random engagement metrics
        views = np.random.randint(10, 1000)
        likes = np.random.randint(0, views + 1)  # Ensure likes don't exceed views
        comments = np.random.randint(0, likes + 1)  # Ensure comments don't exceed likes
        
        # Generate random sentiment score (-1 to 1)
        sentiment = round(np.random.uniform(-1, 1), 2)
        
        conversations.append({
            'id': i+1,
            'content': content,
            'timestamp': date.strftime("%Y-%m-%d %H:%M:%S"),
            'topic': topic,
            'tags': f"{topic},{random.choice(['trending', 'popular', 'new', 'controversial'])}",
            'views': views,
            'likes': likes,
            'comments': comments,
            'sentiment': sentiment,
            'length': len(content.split())  # Word count
        })
    return pd.DataFrame(conversations)

# Generate users
def generate_users(n=50):
    users = []
    for i in range(n):
        join_date = datetime.now() - timedelta(days=np.random.randint(0, 730))
        users.append({
            'id': i+1,
            'username': f"user_{i+1}",
            'join_date': join_date.strftime("%Y-%m-%d")
        })
    return pd.DataFrame(users)

def generate_interactions(users, conversations, n=1000):
    interactions = []
    for _ in range(n):
        user_id = np.random.choice(users['id'])
        conversation = conversations.sample().iloc[0]
        interaction_type = np.random.choice(['view', 'like', 'comment'], p=[0.7, 0.2, 0.1])
        timestamp = datetime.strptime(conversation['timestamp'], "%Y-%m-%d %H:%M:%S") + timedelta(minutes=np.random.randint(0, 1440))
        interactions.append({
            'user_id': user_id,
            'conversation_id': conversation['id'],
            'interaction_type': interaction_type,
            'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'topic': conversation['topic']
        })

    return pd.DataFrame(interactions)

# Generate the datasets
conversations_df = generate_conversations(200)
users_df = generate_users(50)
interactions_df = generate_interactions(users_df, conversations_df, 1000)

# Save to CSV files
conversations_df.to_csv('conversations.csv', index=False)
users_df.to_csv('users.csv', index=False)
interactions_df.to_csv('interactions.csv', index=False)

# Save to CSV
conversations_df.to_csv('conversations.csv', index=False)
print("Enhanced conversations dataset created and saved to conversations.csv")