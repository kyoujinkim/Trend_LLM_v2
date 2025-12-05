"""
Setup Script for Trend Discovery Pipeline
Handles installation, configuration, and environment setup
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def check_python_version():
    """Ensure Python 3.8+ is being used"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")


def create_directory_structure():
    """Create necessary directories for the pipeline"""
    directories = [
        'data',
        'data/news',
        'data/embeddings',
        'data/output',
        'data/output/model',
        'data/output/analysis',
        'data/output/tracking',
        'data/output/visualizations',
    ]

    print("\nCreating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")

    print("✓ Directory structure created")


def create_requirements_file():
    """Create requirements.txt if it doesn't exist"""
    requirements = """# Core dependencies
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0

# Clustering and dimensionality reduction
umap-learn>=0.5.3
hdbscan>=0.8.29
bertopic>=0.15.0

# Data processing
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.10.0

# Visualization
matplotlib>=3.6.0
plotly>=5.14.0
seaborn>=0.12.0

# LLM and API
openai>=1.0.0

# Statistical analysis
scikit-learn>=1.2.0
statsmodels>=0.14.0

# Utilities
tqdm>=4.65.0
python-dateutil>=2.8.0
"""

    if not os.path.exists('requirements.txt'):
        print("\nCreating requirements.txt...")
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        print("✓ requirements.txt created")
    else:
        print("\n✓ requirements.txt already exists")


def install_dependencies(force=False):
    """Install required packages"""
    if not force:
        response = input("\nInstall dependencies from requirements.txt? (y/n): ")
        if response.lower() != 'y':
            print("Skipping dependency installation")
            return

    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        sys.exit(1)


def create_config_template():
    """Create template configuration files"""

    # Project config template
    project_config_template = """[embedding]
model_id = google/embeddinggemma-300M
batch_size = 64
iterate_size = 20480

[data]
data_path = ./data
news_path = ./data/news

[clustering]
min_topic_size = 10
max_topic_size = 10
max_subtopic_size = 5

[keyphraseextract]
model_class = openai
model_id = gpt-4o-mini
max_keyphrase = 5
max_new_tokens = 250
"""

    if not os.path.exists('project_config.ini'):
        print("\nCreating project_config.ini template...")
        with open('project_config.ini', 'w') as f:
            f.write(project_config_template)
        print("✓ project_config.ini created")
    else:
        print("\n✓ project_config.ini already exists")

    # API config template (create at user's home directory)
    api_config_path = Path.home() / 'trend_pipeline_api_config.ini'
    api_config_template = """# API Configuration for Trend Discovery Pipeline
# IMPORTANT: Keep this file secure and never commit to version control

[huggingface]
token = your_huggingface_token_here

[openai]
api_key = your_openai_api_key_here
"""

    if not api_config_path.exists():
        print(f"\nCreating API config template at {api_config_path}...")
        with open(api_config_path, 'w') as f:
            f.write(api_config_template)
        print(f"✓ API config template created")
        print("\n  ⚠ IMPORTANT: Edit this file and add your API keys:")
        print(f"    {api_config_path}")
    else:
        print(f"\n✓ API config already exists at {api_config_path}")


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Data files
data/
*.pkl
*.npy
emb_mapper.json

# API keys and configs
*api_config.ini
D:/config.ini

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Output
output/
visualizations/
"""

    if not os.path.exists('.gitignore'):
        print("\nCreating .gitignore...")
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print("✓ .gitignore created")
    else:
        print("\n✓ .gitignore already exists")


def verify_installation():
    """Verify that key packages are installed"""
    print("\nVerifying installation...")

    packages = [
        'torch',
        'transformers',
        'umap',
        'hdbscan',
        'pandas',
        'numpy',
        'sklearn',
        'plotly',
        'openai'
    ]

    all_installed = True
    for package in packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            all_installed = False

    if all_installed:
        print("\n✓ All key packages verified")
    else:
        print("\n⚠ Some packages are missing. Run setup again with installation.")


def print_next_steps():
    """Print next steps for the user"""
    api_config_path = Path.home() / 'trend_pipeline_api_config.ini'

    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("\n1. Configure API keys:")
    print(f"   Edit: {api_config_path}")
    print("   Add your HuggingFace token and OpenAI API key")

    print("\n2. Add your data:")
    print("   Place CSV files in: data/news/")
    print("   Required columns: class, date, time, source, kind, title, content")

    print("\n3. Update main.py:")
    print("   Change line 366 to point to your API config:")
    print(f"   config.read('{api_config_path}')")

    print("\n4. Run the pipeline:")
    print("   Basic usage: python main.py")
    print("   Help: python main.py --help")

    print("\n5. Explore results:")
    print("   Analysis: python analyze_results.py")
    print("   Explore clusters: python explore_clusters.py")

    print("\n" + "="*80)
    print("\nFor more information, see README.md")
    print("="*80)


def main():
    """Main setup function"""
    print("="*80)
    print("Trend Discovery Pipeline - Setup")
    print("="*80)

    # Check Python version
    check_python_version()

    # Create directory structure
    create_directory_structure()

    # Create requirements file
    create_requirements_file()

    # Ask about installation
    install_dependencies()

    # Create config templates
    create_config_template()

    # Create .gitignore
    create_gitignore()

    # Verify installation
    verify_installation()

    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Setup Trend Discovery Pipeline')
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--force-install', action='store_true',
                       help='Install dependencies without prompting')

    args = parser.parse_args()

    if args.skip_install:
        # Override the install function to skip
        def install_dependencies(force=False):
            print("\nSkipping dependency installation (--skip-install)")
        globals()['install_dependencies'] = install_dependencies
    elif args.force_install:
        # Override to force install
        def install_dependencies(force=False):
            globals()['original_install_dependencies'](force=True)
        globals()['original_install_dependencies'] = install_dependencies
        globals()['install_dependencies'] = install_dependencies

    main()