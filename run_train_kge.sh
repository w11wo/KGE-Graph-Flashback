for city in bandung beijing istanbul jakarta kuwait_city melbourne moscow new_york palembang petaling_jaya sao_paulo shanghai sydney tangerang tokyo; do
    python run_knowledge_representation.py \
        -model_type transe \
        -dataset $city \
        -batch_size 1024 -optimizer_type Adam \
        -version ${city}_scheme2
done