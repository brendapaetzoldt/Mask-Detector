# Mask-Detector

Trabalho desenvolvido com o objetivo de implementar alguma técnica de Inteligência Computacional e demonstrar sua aplicação para a disciplina de Inteligência Computacional do Centro de Educação Superior do Alto Vale do Itajaí (CEAVI/UDESC).

# Equipe
Brenda Paetzoldt - brendapaetzoldt

# Problema
A obrigatoriedade do uso de máscara tem sido cada vez mais comum como método de proteção individual contra o avanço da Covid-19 em locais e vias públicas.
Mesmo com essa obrigatoriedade e necessidade, algumas pessoas ainda se negam a usar uma máscara ao entrar em algum estabelecimento. Cabendo ao responsável pelo lugar 
cobrar o uso pelos frequentadores ou clientes.
Uma simples webcam na entrada de um estabelecimento pode capturar imagens dos frequentadores e identificar o uso correto de máscaras (cobrindo o nariz e boca), libeirando ou não sua entrada ou avisando 
um responsável da presença desta pessoa.

# Dataset
https://github.com/cabani/MaskedFace-Net

67,049 imagens de faces de pessoas usando máscaras corretamente;
66,734 imagens de faces de pessoas usando máscaras incorretamente;

Créditos do Dataset:
    Adnane Cabani, Karim Hammoudi, Halim Benhabiles, and Mahmoud Melkemi, "MaskedFace-Net - A dataset of correctly/incorrectly masked face images in the context of COVID-19", Smart Health, ISSN 2352-6483, Elsevier, 2020, DOI:10.1016/j.smhl.2020.100144
    Karim Hammoudi, Adnane Cabani, Halim Benhabiles, and Mahmoud Melkemi,"Validating the correct wearing of protection mask by taking a selfie: design of a mobile application "CheckYourMask" to limit the spread of COVID-19", CMES-Computer Modeling in Engineering & Sciences, Vol.124, No.3, pp. 1049-1059, 2020, DOI:10.32604/cmes.2020.011663


# Técnica
Utilizei técnicas de Deep Learning com as bibliotecas Dlib e Opencv. Foram desenhadas "bounding boxes" ou retângulos nas máscaras contidas nas imagens do dataset.
A biblioteca Dlib para o treinamento dessas imagens desenhadas.

# Bibliotecas
Dlib é um kit de ferramentas em C ++ que contém algoritmos de Learn Machine e ferramentas para a criação de softwares para resolver problemas do mundo real. 

Opencv é uma biblioteca multiplataforma para o desenvolvimento de aplicativos na área de Visão computacional.


# Instruções para uso do software
Basta clicar no arquivo executável chamado webcam.exe, a webcam irá abrir e detectar se a máscara está sendo utilizada.

# Vídeo
https://udesc-my.sharepoint.com/:v:/g/personal/43900561826_edu_udesc_br/EXEaQWvlI7xMpj6IJt8_454Bqp5NsAlj7nHkEHF7KanCmQ?e=4CLD3k

