from setuptools import setup, find_packages

# This setup.py is kept for compatibility with older tools
# Most configuration is in pyproject.toml
setup(
    packages=find_packages(include=['agentix', 'agentix.*']),
    package_data={
        'agentix': ['py.typed'],  # Include type information
    },
) 