import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain.embeddings import OpenAIEmbeddings
import json
import random

# Initialize session state
if "doc_embeddings" not in st.session_state:
    st.session_state.doc_embeddings = np.load("embeddings.npy")
if "show_labels" not in st.session_state:
    st.session_state.show_labels = False
if "starred_indices" not in st.session_state:
    # Initialize with None, will be set later
    st.session_state.starred_indices = None
if "perplexity" not in st.session_state:
    st.session_state.perplexity = 30
if "learning_rate" not in st.session_state:
    st.session_state.learning_rate = 200

# Load document data
doc_data = json.load(open("harry_docs.json"))

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

st.title("Document Embeddings Visualization")
st.write("Compare base to conditional embeddings")

question_text = st.text_input("Enter a question:")

# Create function for shortened labels
def get_shortened_labels(docs):
    shortened_labels = []
    for doc in docs:
        words = doc.split()
        if len(words) >= 2:
            shortened_label = f"{words[0]} {words[1]}..."
        else:
            shortened_label = doc
        shortened_labels.append(shortened_label)
    return shortened_labels

# Function to select points to be starred
def select_points_to_star(total_points, num_to_star=7):
    """Select random indices to be marked with stars."""
    if num_to_star > total_points:
        num_to_star = total_points
    
    # Generate random indices without duplicates
    indices = set()
    while len(indices) < num_to_star:
        indices.add(random.randint(0, total_points - 1))
    
    return list(indices)

# Create PCA visualization function
def create_pca_visualization(embeddings, doc_data, title, starred_indices, show_labels=False):
    # Apply PCA
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    df = pd.DataFrame(reduced, columns=['PC1', 'PC2'])
    
    # Add document information
    df['Sentence'] = doc_data
    df['ShortLabel'] = get_shortened_labels(doc_data)
    
    # Mark points to be starred
    df['IsStarred'] = False
    if starred_indices is not None:
        df.loc[starred_indices, 'IsStarred'] = True
    
    # Create base figure for non-starred points
    fig = px.scatter(
        df[~df['IsStarred']],
        x='PC1',
        y='PC2',
        hover_data={'Sentence': True, 'ShortLabel': False, 'PC1': False, 'PC2': False},
        color=df[~df['IsStarred']].index.astype(str),
        title=title,
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    # Apply styling based on show_labels for non-starred points
    if show_labels:
        fig.update_traces(
            text=df[~df['IsStarred']]['ShortLabel'],
            textposition='top center',
            textfont=dict(color='black', size=10),
            marker=dict(size=3, line=dict(width=0.5, color='black'))
        )
    else:
        fig.update_traces(
            marker=dict(size=3, line=dict(width=0.5, color='black'))
        )
    
    # Add starred points as a separate trace
    if starred_indices is not None and len(starred_indices) > 0:
        starred_df = df[df['IsStarred']]
        
        # Add star-shaped markers for starred points
        for idx, row in starred_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['PC1']],
                y=[row['PC2']],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='red',
                    line=dict(width=2, color='black')
                ),
                name=f"Star {idx}",
                hovertext=row['Sentence'],
                hoverinfo='text',
                showlegend=False
            ))
            
            # Add labels for starred points if needed
            if show_labels:
                fig.add_trace(go.Scatter(
                    x=[row['PC1']],
                    y=[row['PC2']],
                    mode='text',
                    text=row['ShortLabel'],
                    textposition='top center',
                    textfont=dict(color='black', size=10, family='Arial Black'),
                    showlegend=False
                ))
    
    # Add labels toggle button as a custom shape
    fig.update_layout(
        width=800,
        height=600,
        showlegend=False,
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='#f0f0f0',
        font=dict(color='black', size=12),
        title_font=dict(color='black'),
        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        args=[{"textposition": "top center", 
                               "text": df['ShortLabel'] if not show_labels else None}],
                        label="Toggle Labels",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
                bgcolor='#cccccc',
                font=dict(size=12)
            )
        ]
    )
    
    return fig

# Create t-SNE visualization function 
def create_tsne_visualization(embeddings, doc_data, title, starred_indices, perplexity=30, learning_rate=200, show_labels=False):
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    df = pd.DataFrame(reduced, columns=['TSNE1', 'TSNE2'])
    
    # Add document information
    df['Sentence'] = doc_data
    df['ShortLabel'] = get_shortened_labels(doc_data)
    
    # Mark points to be starred
    df['IsStarred'] = False
    if starred_indices is not None:
        df.loc[starred_indices, 'IsStarred'] = True
    
    # Create base figure for non-starred points
    fig = px.scatter(
        df[~df['IsStarred']],
        x='TSNE1',
        y='TSNE2',
        hover_data={'Sentence': True, 'ShortLabel': False, 'TSNE1': False, 'TSNE2': False},
        color=df[~df['IsStarred']].index.astype(str),
        title=title,
        labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2'},
        color_discrete_sequence=px.colors.qualitative.Vivid
    )
    
    # Apply styling based on show_labels for non-starred points
    if show_labels:
        fig.update_traces(
            text=df[~df['IsStarred']]['ShortLabel'],
            textposition='top center',
            textfont=dict(color='black', size=10),
            marker=dict(size=3, line=dict(width=0.5, color='black'))
        )
    else:
        fig.update_traces(
            marker=dict(size=3, line=dict(width=0.5, color='black'))
        )
    
    # Add starred points as a separate trace
    if starred_indices is not None and len(starred_indices) > 0:
        starred_df = df[df['IsStarred']]
        
        # Add star-shaped markers for starred points
        for idx, row in starred_df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['TSNE1']],
                y=[row['TSNE2']],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=15,
                    color='red',
                    line=dict(width=2, color='black')
                ),
                name=f"Star {idx}",
                hovertext=row['Sentence'],
                hoverinfo='text',
                showlegend=False
            ))
            
            # Add labels for starred points if needed
            if show_labels:
                fig.add_trace(go.Scatter(
                    x=[row['TSNE1']],
                    y=[row['TSNE2']],
                    mode='text',
                    text=row['ShortLabel'],
                    textposition='top center',
                    textfont=dict(color='black', size=10, family='Arial Black'),
                    showlegend=False
                ))
    
    # Add labels toggle button as a custom shape
    fig.update_layout(
        width=800,
        height=600,
        showlegend=False,
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='#f0f0f0',
        font=dict(color='black', size=12),
        title_font=dict(color='black'),
        xaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        yaxis=dict(title_font=dict(color='black'), tickfont=dict(color='black')),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        args=[{"textposition": "top center", 
                               "text": df['ShortLabel'] if not show_labels else None}],
                        label="Toggle Labels",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
                bgcolor='#cccccc',
                font=dict(size=12)
            )
        ]
    )
    
    return fig

# Add t-SNE hyperparameter controls in sidebar
with st.sidebar:
    st.header("t-SNE Parameters")
    st.session_state.perplexity = st.slider("Perplexity", min_value=5, max_value=50, value=30, step=5, 
                                         help="Perplexity is related to the number of nearest neighbors used in t-SNE. Typical values range from 5 to 50.")
    st.session_state.learning_rate = st.slider("Learning Rate", min_value=50, max_value=500, value=200, step=50,
                                            help="Learning rate for t-SNE optimization. Typical values range from 50 to 500.")

# Add a slider to control the number of starred points
num_starred_points = st.sidebar.slider("Number of Starred Points", min_value=3, max_value=20, value=10)

if st.button("Visualize"):
    if question_text:
        # Get query embedding
        query_embedding = embedding_model.embed_query(question_text)
        
        # Select points to be starred if not already set
        if st.session_state.starred_indices is None:
            st.session_state.starred_indices = [i for i in range(len(doc_data) - 10, len(doc_data))]
            print(st.session_state.starred_indices) 
        
        element_wise_mul = query_embedding * st.session_state.doc_embeddings
        
        # Show PCA visualizations
        st.header("PCA Visualizations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Embeddings")
            original_pca = create_pca_visualization(
                st.session_state.doc_embeddings, 
                doc_data, 
                "Original (PCA)",
                st.session_state.starred_indices,
                st.session_state.show_labels
            )
            original_pca.update_layout(height=400, width=400)
            st.plotly_chart(original_pca)
            
        with col2:
            st.write("Conditional Embeddings")
            multiplied_pca = create_pca_visualization(
                element_wise_mul, 
                doc_data, 
                "After Multiplication (PCA)",
                st.session_state.starred_indices,
                st.session_state.show_labels
            )
            multiplied_pca.update_layout(height=400, width=400)
            st.plotly_chart(multiplied_pca)
        
        # Show t-SNE visualizations
        st.header("t-SNE Visualizations")
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("Original Embeddings")
            original_tsne = create_tsne_visualization(
                st.session_state.doc_embeddings, 
                doc_data, 
                "Original (t-SNE)",
                st.session_state.starred_indices,
                st.session_state.perplexity,
                st.session_state.learning_rate,
                st.session_state.show_labels
            )
            original_tsne.update_layout(height=400, width=400)
            st.plotly_chart(original_tsne)
            
        with col4:
            st.write("Conditional Embeddings")
            multiplied_tsne = create_tsne_visualization(
                element_wise_mul, 
                doc_data, 
                "After Multiplication (t-SNE)",
                st.session_state.starred_indices,
                st.session_state.perplexity,
                st.session_state.learning_rate,
                st.session_state.show_labels
            )
            multiplied_tsne.update_layout(height=400, width=400)
            st.plotly_chart(multiplied_tsne)
        
    else:
        st.warning("Please enter a question to visualize the embeddings.")

# Add a checkbox to toggle labels globally
if st.checkbox("Show Labels", value=st.session_state.show_labels):
    st.session_state.show_labels = True
else:
    st.session_state.show_labels = False

