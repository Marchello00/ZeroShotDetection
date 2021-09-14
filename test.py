import requests


def round_struct(struct, digits):
    if isinstance(struct, dict):
        return {k: round_struct(v, digits) for k, v in struct.items()}
    elif isinstance(struct, list):
        return [round_struct(v, digits) for v in struct]
    elif isinstance(struct, float):
        return round(struct, digits)
    else:
        return struct


def test_respond():
    url = "http://0.0.0.0:8088/respond"

    request_data = {
        "queries": {
            "dog.jpg": ['dog', 'кот']
        },
        "image_folder": "."
    }

    result = requests.post(url, json=request_data).json()

    gold_result = {'dog.jpg': [['dog', 36.53, 0.0, 989.4, 1038.18]]}

    digits = 2
    result = round_struct(result, digits)
    gold_result = round_struct(gold_result, digits)
    assert result == gold_result, f"Got\n{result}\n, but expected:\n{gold_result}"
    print("Success")


if __name__ == "__main__":
    test_respond()
