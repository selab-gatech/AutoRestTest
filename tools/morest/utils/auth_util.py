import json
import requests


class SUT:
    SPREE = "SPREE"
    BITBUCKET = 'BITBUCKET'


def spree_login(ip_address):
    spree_payload = {
        "grant_type": "password",
        "username": "spree@example.com",
        "password": "spree123"
    }
    spree_login = requests.post('%s/spree_oauth/token' % str(ip_address), json=spree_payload)
    print(spree_login.text)
    token = "Bearer " + json.loads(spree_login.text)['access_token']
    # token = "Bearer 894fa3111b5a8c4ff8778f2e20f067a367b665918a6eac28"
    return {'Bearer': token}


def bitbucket_login():
    return {"Authorization": "Bearer OTA0NTgxMDU4NzgyOoxSq4aKEaLz4LKmQF3Vhq32ZF2v"}


# this is an util for get access token in multiple sut
# you should pass arguments as following array get_token(sut_name, args) with the same order of the respective handler.
def get_token(*args):
    handlers = {
        SUT.SPREE: spree_login,
        SUT.BITBUCKET: bitbucket_login
    }
    sut_name = args[0]
    if not handlers.__contains__(sut_name):
        return {}
    return handlers[sut_name](*args[1:])
