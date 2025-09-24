import argparse
import json
import requests

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", type=str, default="localhost")
    parser.add_argument("image", type=str, default="localhost")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--mode", type=str, choices=["batch", "stream", "completion"])

    args = parser.parse_args()
    inputs = {
        "input_image": [args.image],
        "input_text": [args.prompt],
        "exmaple_id": ["0"],
    }
    
    if args.mode == "batch":
        postfix = "/batch_generate"
    elif args.mode == "stream":
        postfix = "/completion_stream"
    else:
        postfix = "/completion"

    host = args.host + postfix
    response = requests.post(host, json=inputs, stream=True)
    if response.status_code != 200:
        raise RuntimeError(f"unexpected response: {response}")
    
    if args.mode in ["completion"]:
        output = json.loads(response.content)[0]
        print(f"Output for {output['example_id']}:")
        print(output['prediction'])

    elif args.mode in ["batch"]:
        outputs = json.loads(response.content)
        for output in outputs:
            print(f"Output for {output['example_id']}:")
            print(output['prediction'])

    else:
        for chunk in response.iter_lines(chunk_size=8192, delimiter=b"\0"):
            if chunk:
                chunk = json.loads(chunk.decode("utf-8"))
                content = (
                    chunk["result"]["output"]["text"]
                    if "result" in chunk and "output" in chunk["result"] and "text" in chunk["result"]["output"]
                    else ""
                )

                print(content, end="", flush=True)