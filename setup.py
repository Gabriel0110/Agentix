from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agentix",
    version="0.1.0",
    author="Gabriel Tomberlin",
    author_email="gabrieltomberlin14@gmail.com",
    description="A simple and modular agent framework with a focus on modularity and ease of use. Keep your agentic workflows simple or as complex as you need.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gabriel0110/agentix",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.65.0",
        "pydantic>=2.0.0",
        "aiohttp>=3.8.0",
        "numpy>=1.20.0",
        "python-dotenv>=1.0.0",
        "together>=0.2.8",
        "tavily-python>=0.5.1",
        "duckduckgo-search>=7.5.0",
        "google-genai>=1.3.0"
    ],
) 