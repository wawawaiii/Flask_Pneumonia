import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
import numpy as np
import pydicom as dicom
from fastapi.middleware.cors import CORSMiddleware
import shutil
import base64
import traceback
import torch
import uvicorn

# 업데이트된 모델 코드 임포트
import model_code.my_vgg16 as my_vgg16
import model_code.my_seg as my_seg
import model_code.my_resnet as my_resnet
import model_code.my_alexnet as my_alexnet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

vgg_model = my_vgg16.vgg16_pn()
resnet_model = my_resnet.resnet_pn()
alexnet_model = my_alexnet.alexnet_pn()
seg_model = my_seg.my_seg_model()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/judgePN-fromimg-with-vgg16/")
async def analyze_vgg_image_route(file: UploadFile = File(...)):
    try:
        print("Received request for /judgePN-fromimg-with-vgg16/")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = vgg_model.analyze_image_judgePN(image)
        print(f"VGG16 result: {result}")
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /judgePN-fromimg-with-vgg16/: {e}\n{error_trace}")
        return JSONResponse(status_code=500, content={"error": str(e), "trace": error_trace})



@app.post("/judgePN-fromimg-with-resnet101/")
async def analyze_resnet_image_route(file: UploadFile = File(...)):
    try:
        print("Received request for /judgePN-fromimg-with-resnet101/")
        contents = await file.read()
        print(f"File contents received: {len(contents)} bytes")
        image = Image.open(io.BytesIO(contents))
        result = resnet_model.analyze_image_judgePN(image)
        print(f"ResNet101 result: {result}")
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /judgePN-fromimg-with-resnet101/: {e}\n{error_trace}")
        return JSONResponse(status_code=500, content={"error": str(e), "trace": error_trace})



@app.post("/judgePN-fromimg-with-alexnet/")
async def analyze_alexnet_image_route(file: UploadFile = File(...)):
    print("/judgePN-fromimg-with-alexnet")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = alexnet_model.analyze_image_judgePN(image)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /judgePN-fromimg-with-alexnet/: {e}\n{error_trace}")
        return JSONResponse(status_code=500, content={"error": str(e), "trace": error_trace})


@app.post("/judgePN-fromimg-with-alexnet-web/")
async def analyze_alexnet_image_route(file: UploadFile = File(...)):
    try:
        print("Received request for /judgePN-fromimg-with-alexnet-web/")
        contents = await file.read()
        print(f"File contents received: {len(contents)} bytes")
        file_extension = file.filename.split('.')[-1].lower()

        if file_extension in ['jpg', 'jpeg', 'png']:
            image = Image.open(io.BytesIO(contents))
        elif file_extension == 'dcm':
            # DCM 파일 처리
            dcm = dicom.dcmread(io.BytesIO(contents))
            img_data = dcm.pixel_array
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
            img_data = np.array((img_data * 255), dtype=np.uint8)
            image = Image.fromarray(img_data)
        else:
            return JSONResponse(status_code=400, content={"error": "Invalid file format. Please upload a JPEG, PNG, or DCM file."})

        result = alexnet_model.analyze_image_judgePN(image)
        print(f"AlexNet result: {result}")
        return JSONResponse(status_code=200, content=result)
    except dicom.errors.InvalidDicomError:
        return JSONResponse(status_code=400, content={"error": "Invalid DICOM file. Please upload a valid DICOM file."})
    except PIL.UnidentifiedImageError:
        return JSONResponse(status_code=400, content={"error": "Cannot identify image file. Please upload a valid image file."})
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /judgePN-fromimg-with-alexnet-web/: {e}\n{error_trace}")
        return JSONResponse(status_code=500, content={"error": str(e), "trace": error_trace})

@app.post("/lung-image-mask/")
async def convert_to_mask(file: UploadFile = File(...)):
    print("/lung-image-mask")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        mask_data = seg_model.create_mask(image)
        return StreamingResponse(io.BytesIO(mask_data), media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}

@app.post("/process_dicom_file/")
async def process_dicom_file(file: UploadFile = File(...)):
    print("/process_dicom_file")
    try:
        dcm = dicom.dcmread(file.file)
        img_data = dcm.pixel_array
        output = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        output = np.array((output * 255), dtype=np.uint8)
        dcm_img = Image.fromarray(output)
        output_buffer = io.BytesIO()
        dcm_img.save(output_buffer, format="JPEG")
        return StreamingResponse(io.BytesIO(output_buffer.getvalue()), media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}

@app.post("/judgePN-fromdcm-web/")
async def analyze_dicom_web(file: UploadFile = File(...)):
    print("/judgePN-fromdcm-web")
    try:
        dcm = dicom.dcmread(io.BytesIO(await file.read()))
        img_data = dcm.pixel_array
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        img_data = np.array((img_data * 255), dtype=np.uint8)
        image = Image.fromarray(img_data)
        result = resnet_model.analyze_image_judgePN(image)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return {"image": encoded_image, **result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/analyze-image-with-resnet101/")
async def analyze_image_with_resnet101(file: UploadFile = File(...)):
    print("/analyze-image-with-resnet101")
    try:
        image_data = await file.read()
        if file.filename.endswith('.dcm'):
            image = resnet_model.image_data_processor.process_dicom(image_data)
        else:
            image = Image.open(io.BytesIO(image_data))
            image = resnet_model.image_data_processor.process_image(image)
        tensor = image.unsqueeze(0).to(resnet_model.device)
        output = resnet_model.resnet_model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, preds = torch.max(probabilities, 1)
        class_name = 'PNEUMONIA' if preds.item() == 1 else 'NORMAL'

        # 로그 기록
        print(f"File: {file.filename}, Predicted: {class_name}, Confidence: {confidence.item()}")

        return {"result": class_name, "confidence": confidence.item()}
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /analyze-image-with-resnet101/: {e}\n{error_trace}")
        return JSONResponse(status_code=500, content={"error": str(e), "trace": error_trace})


@app.post("/segment-dicom/")
async def segment_dicom(file: UploadFile = File(...)):
    print("/segment-dicom")
    try:
        with open("model/temp_dicom_file.dcm", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        dcm = dicom.dcmread('model/temp_dicom_file.dcm')
        img_data = dcm.pixel_array
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        img_data = np.array((img_data * 255), dtype=np.uint8)
        image = Image.fromarray(img_data)
        mask_data = seg_model.create_mask(image)
        return StreamingResponse(io.BytesIO(mask_data), media_type="image/jpeg")
    except Exception as e:
        return {"error": str(e)}

@app.post("/judgePN-fromimg/")
async def analyze_vgg_image_route(file: UploadFile = File(...)):
    print("/judgePN-fromimg")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = vgg_model.analyze_image_judgePN(image)
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in /judgePN-fromimg/: {e}\n{error_trace}")
        return JSONResponse(status_code=500, content={"error": str(e), "trace": error_trace})

if __name__ == "__main__":
    uvicorn.run("lung_api_2:app", host='0.0.0.0', port=8000, reload=True)
