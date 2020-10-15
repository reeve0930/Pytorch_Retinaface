import responder
from reelib import vision

from detecter import RetinaDetector

api = responder.API()
retina = RetinaDetector("mobilenet")

@api.route("/")
class RetinaServer:
    async def on_post(self, req, resp):
        data = await req.media()

        img = data.get("image", None)
        if img is None:
            resp.media = {"status":"Error", "message":"You must post a image file."}
            return

        img = vision.decode_img(img)
        results = retina.detect(img)
        for i in range(len(results)):
            results[i]["confidence"] = "{:.8f}".format(results[i]["confidence"])
        resp.media = results

if __name__=="__main__":
    api.run(port=30000, address="0.0.0.0", debug=True)
