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

    text = ["Show me the dog, please!",
            "Найди собаку.",
            "Где здесь кот?"]
    image = ["dog.jpg", "dog.jpg", "dog.jpg"]

    request_data = {
        "text": text,
        "image": image
    }

    result = requests.post(url, json=request_data).json()

    gold_result = {'detected_objects': [
        [
            {
                'l': 27.127676010131836,
                'u': 0.0,
                'r': 1039.8526611328125,
                'd': 1032.8140869140625,
                'label_en': 'Samoyed',
                'label_ru': 'Самоед',
            },
            {
                'l': 386.8840637207031,
                'u': 407.98443603515625,
                'r': 1546.0,
                'd': 1144.8740234375,
                'label_en': 'Great Pyrenees',
                'label_ru': 'Великие Пиренеи',
            }
        ],
        [
            {
                'l': 27.127676010131836,
                'u': 0.0,
                'r': 1039.8526611328125,
                'd': 1032.8140869140625,
                'label_en': 'Samoyed',
                'label_ru': 'Самоед',
            },
            {
                'l': 386.8840637207031,
                'u': 407.98443603515625,
                'r': 1546.0,
                'd': 1144.8740234375,
                'label_en': 'Great Pyrenees',
                'label_ru': 'Великие Пиренеи',
            }
        ],
        []
    ]}

    digits = 2
    result = round_struct(result, digits)
    gold_result = round_struct(gold_result, digits)
    assert result == gold_result, f"Got\n{result}\n, but expected:\n{gold_result}"
    print("Success")


if __name__ == "__main__":
    test_respond()
