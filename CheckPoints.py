from pydantic import BaseModel
from enum import Enum
from typing import List


class CheckPointsEnum(Enum):
    CP1 = [
        "Is it ok to text you on your mobile number?",
    ]

    CP2 = [
        "Nice to have to share an office with somebody like us",
    ]

    CP3 = ["Streets are quite noisy",]
    CP4 = ["My name is John. I will be helping you today.",]
    CP5 = ["Would you like to proceed with enrolling in [plan name] today?",]
    CP6 = [ "In order to serve you better, I'm going to ask you a few questions to verify your identity and get us on the way to your solution",]

'''
# Accessing data:
print(CheckPointsEnum.CP1.value)
for checkpoint in CheckPointsEnum:
    print(checkpoint.name)  # This gives the name of the checkpoint (e.g., "CP1")
    print(checkpoint.value)  # This gives the list of sentences for the checkpoint
'''
