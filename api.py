import binascii
import mimetypes
import os
import time
from datetime import timedelta
from pathlib import Path

import nibabel as nib
import numpy
import SimpleITK as sitk
import torch
import torch.backends.cudnn
import torch.nn.functional as f
from flask import Flask, abort, jsonify, render_template, request, session, Response

app = Flask(__name__)

pprd_init_value = -1
predict_prob_result_dict = {
    "ad": pprd_init_value,
    "nc": pprd_init_value,
}


# 清空结果
def init_result_dict():
    for keynum in predict_prob_result_dict:
        predict_prob_result_dict[keynum] = pprd_init_value


# 保存结果
def save_result_to_dict(result_list):
    init_result_dict()
    dict_keys_list = list(predict_prob_result_dict.keys())
    for keynum in range(0, 2):
        predict_prob_result_dict[dict_keys_list[keynum]] = float(result_list[keynum])


def img_resample(image):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(0)

    new_spacing = [1, 1, 1]
    resample.SetOutputSpacing(new_spacing)
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())

    size = [
        int(
            numpy.round(original_size[0] *
                        (original_spacing[0] / new_spacing[0]))),
        int(
            numpy.round(original_size[1] *
                        (original_spacing[1] / new_spacing[1]))),
        int(
            numpy.round(original_size[2] *
                        (original_spacing[2] / new_spacing[2])))
    ]
    resample.SetSize(size)

    newimage = resample.Execute(image)
    return newimage


# 图片转化为 Tensor 张量
def img_to_tensor(img):
    # img = image.load_img(img_path, target_size=(299, 299))
    data = sitk.ReadImage(img)
    data = img_resample(data)
    data = sitk.GetArrayFromImage(data)
    return torch.from_numpy(data[None, ...][None, ...]).cuda()


# 运行预测
def do_predict(model, img: torch.Tensor):
    with torch.no_grad():
        predicted_result = f.softmax(model(img), dim=-1)
        save_result_to_dict(predicted_result[0])


# ==================================================================================================== #
# Flask 程序块


@app.route("/", methods=["GET"])  # 主页
@app.route("/index.html", methods=["GET"])
def index():
    if not session.get("id"):
        session["id"] = binascii.hexlify(os.urandom(5)).decode()
    return render_template("index.html")


@app.route("/api/v1/predict", methods=["POST"])  # 仅接受 POST 请求
def predict():
    if session.get("id"):
        uid = session["id"]
    else:
        abort(Response("hacker?"))

    upload_file = request.files["image"]
    if upload_file:
        temp_file_path = r"./temp/" + uid + ".nii"
        upload_file.save(temp_file_path)
        data = img_resample(sitk.ReadImage(temp_file_path))
        data = sitk.GetArrayFromImage(data).astype(numpy.float32)
        data = f.normalize(torch.from_numpy(data), p=2, dim=-1).numpy()
        data = torch.from_numpy(data[None, ...][None, ...]).cuda()
        do_predict(model, data)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return jsonify("prob", predict_prob_result_dict)  # 以 Json 序列化的方式返回结果列表
    else:
        abort(400)


if __name__ == "__main__":
    model = torch.load("./best_model.pth").cuda().eval()

    mimetypes.add_type("application/javascript", ".js")
    app.config["SECRET_KEY"] = os.urandom(24)
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=2)
    app.run(host="0.0.0.0", port="9000")
