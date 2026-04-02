O repositório gratuito da FLIR para ADAS (geralmente chamado de FLIR ADAS Thermal Dataset) é bastante usado para treinar modelos de segmentação e detecção com imagens térmicas. Para usar com uma U-Net, você precisa basicamente preparar os dados no formato certo (imagem + máscara) e adaptar o pipeline de treinamento.

O dataset da FLIR normalmente inclui:
- Imagens térmicas (grayscale, mas salvas como RGB às vezes)
- Anotações em bounding boxes (formato tipo COCO)

Pipeline completo:
- Baixar dataset FLIR
- Converter COCO → máscaras
- Organizar imagens/máscaras
- Normalizar imagens térmicas
- Treinar U-Net com Dice Loss
- Avaliar com IoU
