#!usr/bin/bash python

import sys
from google.colab import drive
import subprocess
import os
import shutil

PROJ_NAME = 'ML-Mnemonist'
GIT_DIR = 'HamidrezaKmK'

data_dir = None

if __name__ == '__main__':
  env = os.environ.copy()
  if not os.path.exists('/content/drive'):
    print("Mounting drive...")
    drive.mount('/content/drive')
    print("Mount complete!")

  while True:
    opt = input("What are you trying to do? [clone/pull] ")
    if opt == 'clone':
      addr = f"https://github.com/{GIT_DIR}/{PROJ_NAME}"
      print(f"Trying to connect to {addr}")
      token = input("Enter token: ")
      addr = addr.replace('[TOKEN]', token)
      res = subprocess.run(['git', 'clone', addr], env=env, capture_output=True)
      print(res.stdout.decode())
      print(res.stderr.decode())
      break
    elif opt == 'pull':
      path = os.path.join('/content', PROJ_NAME)
      os.chdir(path)
      res = subprocess.run(['git', 'pull'], env=env, capture_output=True)
      print(res.stdout.decode())
      print(res.stderr.decode())
      break
    elif opt == '':
      print("Nothing happened!")
      break
  
  if not os.path.exists(f'/content/{PROJ_NAME}'):
    raise RuntimeError("No project repository available!")

  if not os.path.exists(f'/content/{PROJ_NAME}/.env'):
    print("Dotenv non-existant!")
    while True:
      resp = input("Do you want to enter the file in the prompt or copy it?\n[copy/write] ")
      if resp == 'copy':
        dir = input("Enter the directory to copy: ")
        shutil.copyfile(dir, f'/content/{PROJ_NAME}/.env')
      elif resp == 'write':
        print("Enter the lines in format ENV_VARIABLE_NAME=VALUE")
        print("End with: ENDFILE")
        with open(f'/content/{PROJ_NAME}/.env', 'w') as f:
          while True:
            line = input()
            if line == 'ENDFILE':
              break
            f.write(f'{line}\n')
      else:
        continue
      break
        
  os.chdir('/content')


