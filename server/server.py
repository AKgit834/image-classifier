from flask import Flask,request,jsonify
import util

app=Flask(__name__)

if __name__ == "__main__":
    print("Starting python Server\n")
    util.load_saved_artifacts()
    ans=util.classify(file_path='D:/vs/jupyter/Source_files/Family_classifier/data/akshita/20230816_235314_006_saved.jpg')
    print(ans)
    app.run(port=5000)