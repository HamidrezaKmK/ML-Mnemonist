rm -rf dist
rm -rf mlmnemonist.egg-info
python3 setup.py sdist
twine upload dist/*