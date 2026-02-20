#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 12:54:38 2022

@author: scotthoran
"""

import pandas as pd 
import requests
import southpaw 
from pandas import DataFrame

basic_auth = 'Basic ZWFmNzdmMTI3ZWEwMDNkNGUyNzVhM2VkMDdkNmY1Mjc6'

fd_email = 'scottshoran@gmail.com'

fd_password = 'Dollar15!'

fd = southpaw.Fanduel(fd_email, fd_password, basic_auth)

up = fd.get_upcoming()

df = DataFrame(up)
