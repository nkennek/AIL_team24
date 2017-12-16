# Google Cloud Vision API取得結果

https://console.cloud.google.com/storage/browser/dena-ai-training-05-gcp/data/hairstyle/goolecvapi_meta/?project=dena-ai-training-05-gcp&organizationId=683655960516

gs://dena-ai-training-05-gcp/data/hairstyle/goolecvapi_meta/
gs://dena-ai-training-05-gcp/data/hairstyle/goolecvapi_meta/hotpepperbeauty.json
gs://dena-ai-training-05-gcp/data/hairstyle/goolecvapi_meta/download_ladys.json

## 元画像

hotpepperbeauty
https://console.cloud.google.com/storage/browser/dena-ai-training-05-gcp/data/hairstyle/hotpepperbeauty/?project=dena-ai-training-05-gcp&organizationId=683655960516

gs://dena-ai-training-05-gcp/data/hairstyle/hotpepperbeauty

download_ladys
https://console.cloud.google.com/storage/browser/dena-ai-training-05-gcp/data/hairstyle/hotpepperbeauty2/?project=dena-ai-training-05-gcp&organizationId=683655960516
download_ladys.zip

# setup
## get_cvapi_json
API_KEYの設定が必要になります。

# usage
## get
```
python ./get_cvapi_json imgfiles.jpg > cv_meta.json
```

## parse
```
python ./parse_cvapi_json cv_meta.json
python ./parse_cvapi_json --debug cv_meta.json
```
