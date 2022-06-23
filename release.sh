rm -rf dist
rm -rf mlmnemonist.egg-info
python3 setup.py sdist
twine upload --repository-url https://upload.pypi.org/legacy/ dist/*