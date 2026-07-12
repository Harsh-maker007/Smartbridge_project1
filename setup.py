"""
setup.py
========
Optional packaging setup for the project.
Allows installation via: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="emotion-learning-platform",
    version="1.0.0",
    description="AI-Driven Emotion Detection & Personalized Learning Support Platform",
    author="Smartbridge AI/ML",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    python_requires=">=3.10",
    install_requires=[
        "tensorflow>=2.15.0",
        "torch>=2.2.0",
        "transformers>=4.38.0",
        "streamlit>=1.32.0",
        "google-generativeai>=0.4.1",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "nltk>=3.8.0",
        "plotly>=5.20.0",
        "python-dotenv>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "train-bilstm=scripts.train_bilstm:main",
            "train-bert=scripts.train_bert:main",
            "run-app=app.main:run",
        ]
    },
)
