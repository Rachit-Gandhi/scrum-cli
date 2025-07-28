from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scrum-cli",
    version="1.0.0",
    author="Rachit Gandhi",
    author_email="rachit@example.com",
    description="AI-powered meeting assistant with real-time transcription and roast mode",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Rachit-Gandhi/scrum-cli",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Groupware",
        "Topic :: Communications :: Conferencing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "click>=8.0.0",
        "chromadb>=0.4.0",
        "fastapi>=0.116.1",
        "google-generativeai>=0.7.0",
        "groq>=0.30.0",
        "numpy==1.26.4",
        "openai-whisper>=20250625",
        "psutil>=5.9.0",
        "pyannote-audio>=3.3.0",
        "pydub>=0.25.1",
        "python-dotenv==1.1.0",
        "requests>=2.28.0",
        "rich>=13.0.0",
        "scikit-learn<1.7.1",
        "selenium>=4.0.0",
        "setuptools-rust>=1.11.1",
        "torch>=2.7.1",
        "torchaudio>=2.7.1",
        "uvicorn>=0.35.0",
        "webdriver-manager>=3.8.0",
        "websockets>=15.0.1",
        "onnxruntime==1.15.0",
        "numba>=0.53.1",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "scrum-cli=scrum_cli.main:cli",
        ],
    },
    include_package_data=True,
    keywords="meeting transcription ai assistant scrum cli roast",
    project_urls={
        "Bug Reports": "https://github.com/Rachit-Gandhi/scrum-cli/issues",
        "Source": "https://github.com/Rachit-Gandhi/scrum-cli",
        "Documentation": "https://github.com/Rachit-Gandhi/scrum-cli/blob/main/README.md",
    },
)
