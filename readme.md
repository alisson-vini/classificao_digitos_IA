# Classificação de digitos usando Pytorch

## Descrição

Esse projeto treina uma rede neural simples que identifica digitos escritos a mão no intervalo de 0-9. A IA foi treinada usando o banco de imagens do MNIST, onde cada uma tem 28x28 pixels, a função de custo utilizada foi a **Cross Entropy Loss** ideal para problemas de classificação onde cada imagem vai pertencer a somente uma classe, o itmizador utilizado foi o **SGD**, testei também com o Adam mas obtive desempenho inferior.

## Como rodar

Para rodar o codigo é preciso ter o python, Pillow e pytoch instalado, caso queira testar alguma imagem de um digito escrito a mão por voce a imagem precisa ter fundo branco (digito escrito no papel branco), corte a imagem para deixar o digito mais ou menos centralizado como no exemplo "numero_8" presente nessa pasta, coloque a imagem dentro da pasta do projeto, na linha 93 do código coloque o diretório da imagem; Apos isso basta rodar o código.

caso não tenha o Pillow ou pytorch instalado coloque no CMD:
pip install torch torchvision pillow

a saida esperada deve ser algo como:

    # Erros ao longo do treinamento
    erro da epoca 1 - 120.45
    erro da epoca 2 - 49.32
    erro da epoca 3 - 33.92
    erro da epoca 4 - 25.47
    erro da epoca 5 - 21.02

    # Previsão da imagem
    O numero na imagem é 8

    # Acurácia do modelo depois do treino
    Acurácia na época 5: 97.58%

## problemas enfrentados

O principal problema dessa rede é que ela esta com grande dificuldade de conseguir identificar fotos reais, tiradas pelo celular por exemplo, acredito que isso se deve ao banco de fotos do MNIST que segue um padrão especifico de imagens (centralizadas, branco e preto, luminosidade...), quando a rede lida com fotos reais quase nunca as fotos são nesse formato exato, por essa razão a rede erra constantemente, acredito que uma forma de resolver isso seria a implementação de uma rede neral convolusional para conseguir extrair as principais caracteristicas da imagem antes de classifica-la.