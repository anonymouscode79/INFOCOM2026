import json
def check_serializable(obj):
    try:
        json.dumps(obj)
        print("Serializable")
    except TypeError as e:
        print(f"Not serializable: {e}")

label_results = {'1': [[0], [76]]}  # Replace with your actual data
check_serializable(label_results)