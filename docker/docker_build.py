#!/usr/bin/env python3
import os

if __name__=="__main__":
    cmd = "docker build -t msphinx . "
    code = os.system(cmd)
