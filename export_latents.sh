pip install safetensors huggingface_hub wandb

tar -zxf embryo_dataset.tar.gz
tar -zxf latents.tar.gz
tar -zxf embryo_dataset_annotations.tar.gz
if [ -f "api_keys.txt" ]; then
    HF_KEY=$(head -n 1 api_keys.txt)
    export HF_TOKEN=$HF_KEY
fi
ls -lh
IFS="_" read -ra ADDR <<< "$1"

python export_latents.py --name notemp-2026-02-20
python export_latents.py --name control-2026-02-20
python export_latents.py --name noconv-2026-02-20

mkdir -p latents
mv *.npy latents/
mv *.csv latents/
tar -I 'gzip -1' -cf latents.tar.gz latents/

