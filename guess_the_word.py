import streamlit as st
import json
import random
from collections import defaultdict
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import os
import re

st.set_page_config(
    page_title="Guess the word",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

wrong_answer_prefixes = [
        "Ah... the right word was ",
        "You're fired. The right word was "
        "No. The best word was "
        "Hmm... we were looking for ",
        "Oh dear, the correct word is ",
        "Actually, we meant ",
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

def load_corpus(file_path):
    """Load and parse JSON corpus"""
    with open(file_path, 'r') as f:
        return json.load(f)


def analyze_emotions_across_corpus(corpus_sentences, keyword, replacement_word):
    """Analyze emotional impact across all sentences containing the keyword"""
    emotions_original = []
    emotions_modified = []
    example_pairs = []

    analyzer = SentimentIntensityAnalyzer()

    # Analyze each sentence containing the keyword
    for sentence_data in corpus_sentences:
        if keyword.lower() in sentence_data['sentence'].lower():
            original = sentence_data['sentence']
            modified = original.replace(keyword, replacement_word)

            orig_emotions = analyzer.polarity_scores(original)["compound"]
            mod_emotions = analyzer.polarity_scores(modified)["compound"]

            emotions_original.append(orig_emotions)
            emotions_modified.append(mod_emotions)

            total_emotion_shift = (sum(emotions_modified) - sum(emotions_original))/(len(emotions_modified) + len(emotions_original))
            example_pairs.append({
                'original': original,
                'modified': modified,
                'emotion_shift': total_emotion_shift
            })

    # Calculate averages
    avg_original = sum(emotions_original)/len(emotions_original)
    avg_modified = sum(emotions_modified)/len(emotions_modified)

    # Create chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value= - (avg_modified - avg_original),
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [-1, 0], 'color': 'darkgrey'},
                {'range': [0, 1], 'color': 'lightgrey'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': - avg_original}
        },
        title={'text': "Darkside-O-Meter"}
    ))

    fig.update_layout(height=300)


    return {
        'chart': fig,
        'example_pairs': sorted(example_pairs, key=lambda x: x['emotion_shift'], reverse=True)
    }


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


def main():
    st.title("Guess the word and deal with the consequences")

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

    keywords = ['economy', 'border', 'death penalty', 'drugs', 'migrants', 'love', 'disaster']

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
                # Analyze impact across corpus
                impact_analysis = analyze_emotions_across_corpus(
                    st.session_state.used_sentences + [st.session_state.current_sentence_data],
                    st.session_state.current_sentence_data['keyword'],
                    user_guess
                )

                # Show example of impact in another sentence
                other_sentences = get_sentences_for_keyword(st.session_state.corpus,
                                                            st.session_state.current_sentence_data['keyword'])
                other_sentences = [s for s in other_sentences
                                   if s['sentence'] != st.session_state.current_sentence_data['sentence']]

                if other_sentences:
                    example = random.choice(other_sentences)
                    colw.markdown("Looks like you also said:")
                    players_sentence = example['sentence'].replace(
                            st.session_state.current_sentence_data['keyword'],
                            user_guess
                        )
                    colw.markdown("> #### {}".format(players_sentence))

                # Show emotion comparison chart
                colw.plotly_chart(impact_analysis['chart'])
                with colw.popover("i"):
                    st.markdown("""
                    The number indicates the change in darkness between the original version and your version. 
                    The gauge shows where the original sentence stands with a tick mark, 
                    and where the modified sentence stands with the gauge filling, on a scale to bright side (-1) to dark side (+1), 
                    
                    This sentiment analysis uses the python vaderSentiment library.""")

            # Add current sentence to used list
            st.session_state.used_sentences.append(st.session_state.current_sentence_data)

        # Next round button
        if st.session_state.submitted:
            if colq.button("Next Round", key=f"next_{st.session_state.current_round}"):
                st.session_state.current_round += 1
                st.session_state.current_sentence_data = None
                st.session_state.submitted = False
                st.rerun()


if __name__ == "__main__":
    main()