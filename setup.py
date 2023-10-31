from setuptools import find_packages, setup

setup(
    name="llm4bi_embedder",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "lightning",
        "transformers",
        "pandas",
        "psycopg2-binary",
        "google-cloud-storage",
        "wandb",
        "pyarrow",
        "fastparquet",
        "sqlalchemy",
    ],
    author="Luis Garcia",
    author_email="admin@huginns.io",
    description="Package including any embedder for the LLM4BI project",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="GNU GPLv3",
    url="git@github.com:ottersome/patent_embedder.git",
)
