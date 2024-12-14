import os

import streamlit as st
import json
import random
from collections import defaultdict
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

st.set_page_config(
    page_title="Guess the word",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

wrong_answer_prefixes = [
        "Ah. the right word was ",
        "You're fired. The right word was "
        "No. The best word was "
        "Hmm... we were looking for ",
        "Oh dear, the correct word is ",
        "Just a bit off! It's ",
        "Let me help - it's ",
        "Here's a hint: it's ",
        "Sorry to say, but it's ",
        "Aw, So ClOsE! It's ",
        "Just a small correction - it's ",
        "Fun fact: the answer is ",
        "Plot twist! It's actually ",
        "Surprise! The word is ",
        "Oh snap! The answer is ",
        "Would you believe it's ",
        "Drum roll please... it's ",
        "Ta-da! The answer is ",
        "Allow me to clarify - it's ",
        "Actually, we're looking for ",
        "Interesting guess, but it's ",
        "Just between us, it's ",
        "Here's the scoop - it's ",
        "Ready for this? It's ",
        "Spoiler alert: it's ",
        "The truth is... it's ",
        "Here's a secret - it's ",
        "You might be surprised, but it's ",
        "Guess what? It's ",
        "The actual word is ",
        "Let me tell you - it's ",
        "Here's the deal: it's ",
        "Believe it or not, it's ",
        "Fun surprise! It's ",
        "Wait for it... it's ",
        "Just FYI - it's ",
        "Quick correction: it's ",
        "Tiny detail - it's ",
        "Minor adjustment: it's ",
        "Slight change - it's ",
        "Little secret: it's ",
        "Pro tip: it's ",
        "Inside scoop: it's ",
        "Breaking news: it's ",
        "Update: it's ",
        "Newsflash: it's ",
        "For reference, it's ",
        "Just so you know, it's ",
        "Here's the thing: it's ",
        "The key is... it's ",
        "Word to the wise: it's ",
        "Picture this: it's ",
        "Fun fact time: it's ",
        "Exciting news! It's ",
        "Big reveal: it's ",
        "Between friends, it's ",
        "Take note - it's ",
        "Minor detail: it's ",
    ]

def tokenize(text):
    """
    Simple tokenizer that:
    - Splits on whitespace and punctuation
    - Preserves contractions
    - Converts to lowercase
    - Removes special characters
    """
    # Clean the text
    text = text.lower()
    # Split on whitespace and punctuation, but preserve words with apostrophes
    tokens = re.findall(r"[a-z0-9]+(?:[''][a-z]+)?", text)
    return tokens

def split_into_sentences(text):
    """
    Split text into sentences using regex.
    Handles common sentence endings (., !, ?) and common abbreviations
    """
    # Add spaces around punctuation marks for easier splitting
    text = re.sub(r'([.!?])', r' \1 ', text)
    # Handle common abbreviations (Mr., Dr., etc.)
    text = re.sub(r'\s+([Mm]r|[Dd]r|[Pp]rof|[Mm]rs|[Mm]s|[Jj]r|[Ss]r|[Ss]t)\.\s+', r' \1$$$', text)
    # Split on sentence endings
    sentences = re.split(r'\s*[.!?]\s+', text)
    # Clean up and restore abbreviations
    sentences = [s.strip().replace('$$$', '.') for s in sentences if s.strip()]
    return sentences


def measure_corpus_similarity(corpus, new_sentence, verbose=False):
    """
    Measures how well a new sentence fits within a given text corpus using TF-IDF similarity.
    """
    if not corpus or not new_sentence:
        raise ValueError("Corpus and new_sentence cannot be empty")

    # Custom tokenizer to ensure we capture all words
    def custom_tokenizer(text):
        # Convert to lowercase and split on whitespace
        words = text.lower().split()
        # Remove any non-alphanumeric characters from each word
        words = [re.sub(r'[^\w\s]', '', word) for word in words]
        # Remove empty strings
        words = [word for word in words if word]
        return words

    if verbose:
        print("\nDebug Information:")
        print(f"Corpus size: {len(corpus)} documents")
        print(f"New sentence: '{new_sentence}'")
        print(f"Words in new sentence: {len(custom_tokenizer(new_sentence))}")
        print("Sample words from new sentence:", custom_tokenizer(new_sentence)[:5])

    # Create and fit TF-IDF vectorizer with diagnostic parameters
    vectorizer = TfidfVectorizer(
        tokenizer=custom_tokenizer,
        lowercase=True,  # Convert everything to lowercase
        stop_words=None,  # Keep all words
        token_pattern=None,  # Use our custom tokenizer instead
        min_df=1,  # Keep terms that appear at least once
        max_df=1.0,  # Keep all terms regardless of frequency
        ngram_range=(1, 2),  # Use only unigrams
        smooth_idf=True,  # Apply smoothing to IDF weights
    )

    # Add the new sentence to the corpus temporarily for vectorization
    all_texts = corpus + [new_sentence]

    # Transform texts to TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    if verbose:
        # Print detailed vocabulary information
        print(f"\nVocabulary size: {len(vectorizer.vocabulary_)}")
        print("Sample vocabulary terms:", list(vectorizer.vocabulary_.keys())[:5])
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Get the terms in the new sentence that are in the vocabulary
        new_sentence_tokens = set(custom_tokenizer(new_sentence))
        vocab_terms = set(vectorizer.vocabulary_.keys())
        common_terms = new_sentence_tokens.intersection(vocab_terms)
        print(f"\nWords from new sentence found in vocabulary: {len(common_terms)}")
        print("Sample matching words:", list(common_terms)[:5])

        # Print the non-zero features in the new sentence vector
        new_sentence_vector = tfidf_matrix[-1]
        non_zero_indices = new_sentence_vector.nonzero()[1]
        feature_names = vectorizer.get_feature_names_out()
        print("\nNon-zero features in new sentence:")
        for idx in non_zero_indices:
            print(f"{feature_names[idx]}: {new_sentence_vector[0, idx]:.3f}")

    # Get TF-IDF similarities
    new_sentence_vector = tfidf_matrix[-1:]
    corpus_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(new_sentence_vector, corpus_vectors)[0]

    if verbose:
        print("\nTop 5 most similar documents:")
        sorted_indices = np.argsort(similarities)[::-1][:5]
        for idx in sorted_indices:
            sim = similarities[idx]
            if sim > 0:
                print(f"Doc {idx}: {sim:.3f} - '{corpus[idx][:50]}...'")
            else:
                print(f"Doc {idx}: {sim:.3f} - '{corpus[idx][:50]}...' (no overlap)")

    # Calculate metrics
    avg_similarity = float(np.mean(similarities))
    max_similarity = float(np.max(similarities))
    most_similar_idx = np.argmax(similarities)

    metrics = {
        'average_similarity': avg_similarity,
        'max_similarity': max_similarity,
        'most_similar_sentence': corpus[most_similar_idx],
        'similarity_distribution': {
            'std': float(np.std(similarities)),
            'median': float(np.median(similarities)),
            'percentiles': {
                '25th': float(np.percentile(similarities, 25)),
                '75th': float(np.percentile(similarities, 75))
            }
        }
    }

    return avg_similarity, metrics


def plot_corpus_similarity(corpus1_sim, corpus2_sim, corpus1_name="MAGA", corpus2_name="DEMOCRAT"):
    """
    Creates an interactive visualization showing where a sentence falls between two corpora
    based on similarity scores, with a split-colored bar.

    Parameters:
    corpus1_sim (float): Similarity score with first corpus (0-1)
    corpus2_sim (float): Similarity score with second corpus (0-1)
    corpus1_name (str): Name of the first corpus
    corpus2_name (str): Name of the second corpus
    """

    # Convert to percentages
    sim1 = round(corpus1_sim * 100, 1)
    sim2 = round(corpus2_sim * 100, 1)

    # Calculate relative position (weighted average)
    total_sim = sim1 + sim2
    position = 50 if total_sim == 0 else (sim2 / total_sim) * 100

    # Create the figure
    fig = go.Figure()

    # Add left half of the bar (Corpus 1)
    fig.add_trace(go.Scatter(
        x=[0, 50],
        y=[0, 0],
        mode='lines',
        line=dict(
            color='rgb(239, 68, 68)',  # Blue (239, 68, 68)
            width=20,
        ),
        hoverinfo='skip'
    ))

    # Add right half of the bar (Corpus 2)
    fig.add_trace(go.Scatter(
        x=[50, 100],
        y=[0, 0],
        mode='lines',
        line=dict(
            color='rgb(59, 130, 246)',  # Red
            width=20,
        ),
        hoverinfo='skip'
    ))

    # Add central dividing line
    fig.add_trace(go.Scatter(
        x=[50, 50],
        y=[-0.2, 0.2],
        mode='lines',
        line=dict(
            color='black',
            width=2,
        ),
        hoverinfo='skip'
    ))

    # Add marker for position
    fig.add_trace(go.Scatter(
        x=[position],
        y=[0],
        mode='markers',
        marker=dict(
            size=25,
            symbol='circle',
            color='white',
            line=dict(
                color='black',
                width=2
            )
        ),
        hoverinfo='skip'
    ))

    # Add labels for corpora
    fig.add_annotation(
        x=25,
        y=-0.5,
        text=f"{corpus1_name}",
        showarrow=False,
        font=dict(size=20)
    )
    fig.add_annotation(
        x=75,
        y=-0.5,
        text=f"{corpus2_name}",
        showarrow=False,
        font=dict(size=20)
    )

    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-10, 110]
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            range=[-1, 1]
        ),
        margin=dict(l=20, r=20, t=20, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=150,
    )

    return fig


def load_corpus(file_path):
    """Load and parse JSON corpus"""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_sentences_for_keyword(corpus, keyword):
    """Find all sentences containing the given keyword"""
    sentences = []
    for speech in corpus['content']:
        for sentence in split_into_sentences(speech['content']):
            if keyword.lower() in sentence.lower():
                sentences.append({
                    'sentence': sentence,
                    'author': speech['author'],
                    'context': corpus['context'],
                    'date': corpus['date'],
                    'keyword': keyword
                })

    return sentences


def get_unused_sentence(used_sentences, corpus, keywords):
    """Get a random sentence for a random keyword"""
    # First, randomly select a keyword
    keyword = random.choice(keywords)

    # Get all sentences containing this keyword
    available_sentences = get_sentences_for_keyword(corpus, keyword)

    # Remove used sentences
    used_set = {s['sentence'] for s in used_sentences}
    available_sentences = [s for s in available_sentences if s['sentence'] not in used_set]

    if not available_sentences:
        # If no sentences available for this keyword, try another
        remaining_keywords = [k for k in keywords if k != keyword]
        if not remaining_keywords:
            return None
        return get_unused_sentence(used_sentences, corpus, remaining_keywords)

    return random.choice(available_sentences)


import plotly.graph_objects as go
import streamlit as st
import random


def get_sarcastic_comment(dem_score, maga_score):
    """
    Generate a sarcastic comment based on relative similarity scores.
    """
    total = dem_score + maga_score
    if total < 0.03:
        neutral_comments = [
            "Perfectly balanced, as all things should be...",
            "Switzerland called, they want their neutrality back",
            "Standing so firmly in the middle you might as well be a traffic cone",
            "Found the centrist!",
            "Ah yes, the rare political chameleon"
        ]
        return random.choice(neutral_comments)

    # Calculate lean (positive for MAGA, negative for Democrat)
    lean = (maga_score - dem_score) / total

    comments = {
        'strong_dem': [
            "you communist!",
            "socialist!",
            "AOC would be proud!",
            "Did Bernie write this?",
            "Marx called, he wants his manifesto back!",
            "Found the avocado toast enthusiast!",
            "Someone's been reading too much Mother Jones",
            "Elizabeth Warren just liked this",
            "Peak coastal elite energy",
            "Living that blue state life",
            "Warning: May contain traces of universal healthcare",
            "Green New Deal energy detected",
            "Squad member spotted!",
            "⚠️ Warning: High levels of progressive thinking detected",
            "Someone's been to one too many climate protests",
            "Whole Foods shopping energy",
            "Spotted the NPR subscriber",
            "Rachel Maddow would approve",
            "Found the plant-based political statement",
            "Democratic party wants to know your location"
        ],
        'mild_dem': [
            "that sounds more democrat",
            "looks more democrat",
            "slight blue wave detected",
            "someone's been watching MSNBC",
            "Moderate Democrat has entered the chat",
            "Blue-curious detected",
            "Slight liberal tendencies observed",
            "Left-leaning, but still pays for Netflix",
            "Biden's speechwriter, is that you?",
            "Centrist with a blue tint",
            "Mildly progressive thoughts detected",
            "CNN+ subscriber (for that one week)",
            "Pete Buttigieg energy",
            "Has definitely used the word 'bipartisan'",
            "Probably owns a hybrid car",
            "Displays mild symptoms of liberalism",
            "Occasionally reads The New York Times",
            "Democrat-ish™",
            "Blue dog with a moderate bite",
            "Likely owns a 'Nevertheless, she persisted' mug"
        ],
        'neutral': [
            "playing both sides, eh?",
            "trying to stay neutral?",
            "fence-sitting level: expert",
            "diplomatic immunity activated",
            "The Switzerland of sentences",
            "Political chameleon detected",
            "Walking the purple line",
            "Schrödinger's political stance",
            "Bipartisan vibes intensify",
            "Found the swing voter!",
            "Both parties are typing...",
            "Perfectly balanced, as all things should be",
            "The rare purple unicorn appears",
            "Master of the political tightrope",
            "Equal opportunity criticizer",
            "Alert: Dangerous levels of centrism detected",
            "Political Switzerland has entered the chat",
            "Peak fence-sitting achievement unlocked",
            "Quantum political superposition achieved",
            "The elusive moderate has been spotted"
        ],
        'mild_maga': [
            "that sounds more MAGA",
            "red pill detected",
            "Fox News subscriber spotted",
            "slight red wave detected",
            "Newsmax browser history found",
            "Getting warmer... and redder",
            "Conservative-curious energy",
            "Right-leaning tendencies observed",
            "Mild case of MAGA detected",
            "Tucker Carlson viewing history detected",
            "Red hat curious",
            "Probably owns a pickup truck",
            "Has strong opinions about pronouns",
            "Mildly spicy conservative takes",
            "Right-curious but still uses TikTok",
            "Watches Fox News, but only Tucker",
            "Has opinions about Hunter's laptop",
            "Conservative with a moderate twist",
            "Right-leaning, but still drinks Starbucks",
            "Has retweeted Dan Bongino at least once"
        ],
        'strong_maga': [
            "MAGA intensifies!",
            "Trump's twitter ghost, is that you?",
            "Make Sentences Great Again!",
            "Someone's been to a rally!",
            "Red wave incoming!",
            "Let's Go Brandon energy detected",
            "Found the wall enthusiast!",
            "Peak patriot energy",
            "Freedom intensifies! 🦅",
            "Conservative bingo winner!",
            "America First vocabulary detected",
            "Red hat energy maximum",
            "Stop the steal vibes",
            "Deep State conspiracy theorist spotted",
            "Peak Newsmax energy",
            "Truth Social's top contributor",
            "Mar-a-Lago VIP detected",
            "Probably has strong opinions about emails",
            "Definitely has an opinion about Dr. Fauci",
            "Professional liberal triggerer"
        ]
    }

    # Select comment category based on lean
    if lean < -0.1:
        category = 'strong_dem'
    elif lean < 0:
        category = 'mild_dem'
    elif lean == 0:
        category = 'neutral'
    elif lean <= 0.1:
        category = 'mild_maga'
    else:
        category = 'strong_maga'

    return random.choice(comments[category])


def main():
    st.title("Guess the word")

    # Initialize session state
    if 'score' not in st.session_state:
        st.session_state.score = 0
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 0
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    if 'corpus' not in st.session_state:
        st.session_state.corpus = None
    if 'used_sentences' not in st.session_state:
        st.session_state.used_sentences = []
    if 'current_sentence_data' not in st.session_state:
        st.session_state.current_sentence_data = None
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False
    if 'author'not in st.session_state:
        st.session_state.author = None
    if "trump_corpus" not in st.session_state:
        st.session_state["trump_corpus"] = []
        for corpus_file in [
            "./corpus/authors?trump_date?10232024_context?rally_duluth_georgia.json",
            "./corpus/authors?Trump_date?11022020__context?campaign_rally_Grand_Rapid.json"
        ]:
            with open(corpus_file, "r") as f:
                c = json.load(f)
                st.session_state["trump_corpus"] += split_into_sentences(c["content"][0]["content"])
            st.session_state.author = c["content"][0]["author"]
    if "democrat_corpus" not in st.session_state:
        st.session_state["democrat_corpus"] = []
        for corpus_file in [
            "./corpus/authors?harris_date?10302024__context?closing_remarks.json",
            "./corpus/authors?biden_date?11022020__context?campaign_rally.json",
            "./corpus/authors?harris_date?11062024_context?acceptance_speech.json"
        ]:
            with open(corpus_file, "r") as f:
                c = json.load(f)
                st.session_state["democrat_corpus"] += split_into_sentences(c["content"][0]["content"])

    keywords = ['economy', 'border', 'death penalty', 'drugs', 'migrant', 'love', 'disaster', 'tariff', "vaccine"]

    # Game progress
    if st.session_state.corpus:
        total_sentences = sum(len(get_sentences_for_keyword(st.session_state.corpus, k)) for k in keywords)
        st.progress(len(st.session_state.used_sentences) / total_sentences, "Progress over available sentences")

    # Game interface
    if not st.session_state.game_active:
        if st.button("Start New Game"):
            try:
                st.session_state.corpus = load_corpus(
                    "./corpus/authors?trump_date?12052024__context?meet_the_press.json")
                st.session_state.used_sentences = []
                st.session_state.game_active = True
                st.session_state.current_round = 1
                st.session_state.submitted = False
                st.session_state.current_sentence_data = None
                st.rerun()
            except Exception as e:
                st.error(f"Error loading corpus: {str(e)}")
    else:
        if not st.session_state.current_sentence_data:
            new_sentence_data = get_unused_sentence(
                st.session_state.used_sentences,
                st.session_state.corpus,
                keywords
            )

            if new_sentence_data is None:
                st.success(f"Game Over! Final Score: {st.session_state.score}")
                if st.button("Start New Game"):
                    st.session_state.used_sentences = []
                    st.session_state.current_round = 1
                    st.session_state.score = 0
                    st.session_state.current_sentence_data = None
                    st.session_state.submitted = False
                    st.rerun()
                return

            st.session_state.current_sentence_data = new_sentence_data

        colq, colw = st.columns(2)
        # Display current round
        colq.markdown("{} at {}, {}".format(st.session_state.corpus["authors"][0],
                                           st.session_state.corpus["context"],
                                           st.session_state.corpus["date"]))
        colq.markdown("#### Round {}; Guess the word!".format(st.session_state.current_round))

        # Display sentence with blank
        sentence_with_blank = st.session_state.current_sentence_data['sentence'].replace(
            st.session_state.current_sentence_data['keyword'],
            "_____"
        ).replace(st.session_state.current_sentence_data['keyword'].capitalize(),
            "_____")
        colq.markdown("> #### {}".format(sentence_with_blank))

        # Get user input
        user_guess = colq.text_input("Your guess:", key=f"guess_{st.session_state.current_round}")

        # Submit button
        if colq.button("Submit", key=f"submit_{st.session_state.current_round}") and not st.session_state.submitted:
            st.session_state.submitted = True

            # Check if guess is correct
            if user_guess.lower() == st.session_state.current_sentence_data['keyword'].lower():
                st.session_state.score += 10
                colw.success("Correct!")
            else:
                colw.markdown("#### " + random.choice(wrong_answer_prefixes) + '"{}"'.format(st.session_state.current_sentence_data['keyword']))


                impact_analysis, metrics = measure_corpus_similarity(st.session_state["trump_corpus"],
                                                                     st.session_state.current_sentence_data['sentence']
                                                                     .replace( st.session_state.current_sentence_data['keyword'], user_guess)
                                                                     )
                #st.write("trump metrics: {}".format(metrics))
                demo_analysis, demo_metrics = measure_corpus_similarity(st.session_state["democrat_corpus"],
                                                                     st.session_state.current_sentence_data['sentence']
                                                                     .replace( st.session_state.current_sentence_data['keyword'], user_guess)
                                                                     )
                #st.write("democrat metrics: {}".format(demo_metrics))

                # show graph

                fig = plot_corpus_similarity(metrics["max_similarity"], demo_metrics["max_similarity"])
                st.plotly_chart(fig, use_container_width=True)


                # comment
                comment = get_sarcastic_comment(demo_metrics["max_similarity"], metrics["max_similarity"])
                st.markdown("#### ***{}***".format(comment))



                # Show example of impact in another sentence
                other_sentences = get_sentences_for_keyword(st.session_state.corpus,
                                                            st.session_state.current_sentence_data['keyword'])
                other_sentences = [s for s in other_sentences
                                   if s['sentence'] != st.session_state.current_sentence_data['sentence']]

                if other_sentences:
                    example = random.choice(other_sentences)
                    colw.markdown("With this change, he would also have said: ")
                    players_sentence = example['sentence'].replace(
                            st.session_state.current_sentence_data['keyword'],
                            user_guess
                        )
                    colw.markdown("> #### {}".format(players_sentence))

                colw.markdown("It reminds me when during a 2024 rally he said")
                colw.markdown("> #### {}".format(metrics["most_similar_sentence"]))

            # Add current sentence to used list
            st.session_state.used_sentences.append(st.session_state.current_sentence_data)

        # Next round button
        if st.session_state.submitted:
            if st.button("Next Round", key=f"next_{st.session_state.current_round}"):
                st.session_state.current_round += 1
                st.session_state.current_sentence_data = None
                st.session_state.submitted = False
                st.rerun()


if __name__ == "__main__":
    main()