import PIL.Image
import PIL.ImageOps
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL

"""
    O precesso de pre-processamento de dados ajuda a rede neural a aprender melhor, evitar desaparecimento ou explosão do gradiente e faz
    com que a rede não de mais "importancia" para valores numericamente maiores.
"""

transform = transforms.Compose([    # operação de pre-processamento de dados que vai ser feita em cada uma das imagens importadas do MNIST
    transforms.ToTensor(),          # serve para normalizar as imagens que estão com valores entre 0-255 para valores entre 0-1
    transforms.Normalize((0.1307,), (0.3081,))
])



# Importa o conjunto de imagens do MNIST e estrutura os Batchs (lotes de imagens)

# Dataset de treino e de teste do MNIST
dataset_treino = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataset_teste = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)



# Construção dos batchs de treino e teste
batch_treino = torch.utils.data.DataLoader(dataset_treino, batch_size=64, shuffle=True)
batch_teste = torch.utils.data.DataLoader(dataset_teste, batch_size=64, shuffle=False)



# Construção da rede neural
class Rede(nn.Module):
    def __init__(self):
        super(Rede, self).__init__()

        self.c1 = nn.Linear(28*28, 128) # entrada de 28*28, pois as imagens do MNIST tem 28 pixels de altura e 28 pixels de largura
        self.c2 = nn.Linear(128, 64)
        self.c3 = nn.Linear(64, 10)     # saida de 10 valores, um para cada digito que a rede vai classificar (0,1,2,3...9)

    def forward(self, entrada):
        entrada = entrada.view(-1, 28*28)      # flatter da imagem
        entrada = torch.relu(self.c1(entrada)) # as entradas passam pela 1° camada e depois a função de ativação relu é aplicada
        entrada = torch.relu(self.c2(entrada)) # as entradas passam pela 2° camada e depois a função de ativação relu é aplicada
        entrada = self.c3(entrada)             # a ultima camada da rede não tem função de ativação, retorna os logits
        return entrada
    

# Definição da função de custo e otimizador
modelo = Rede()
funcao_custo = nn.CrossEntropyLoss() # define a função de custo, a Cross Entropy Loss é ideal para classificação
otimizador = torch.optim.SGD(modelo.parameters(), lr=0.01, momentum=0.9) # defini o otimizador usado para atualização dos pesos e a taxa de aprendizado


# treinamento da rede
epocas = 5

for epoca in range(epocas):

    erro_epoca = 0 # erro acumulado por epoca
    modelo.train() # indica que é o treinamento da rede

    for imagem, label in batch_treino:
        otimizador.zero_grad() # zera o gradiente

        hipotese = modelo(imagem) # calcula a hipotese
        erro = funcao_custo(hipotese, label) # calcula o erro
        erro_epoca += erro.item()
        erro.backward() # calcula o gradiente (derivada da função de custo em relação ao peso/vies) para cada neurôino
        otimizador.step() # atualiza os pesos e bias usando o otimizador escolhido

    print(f"erro da epoca {epoca+1} - {erro_epoca:.2f} ")



# teste do modelo

modelo.eval()
acertos = 0
total = 0

with torch.no_grad():  # Desativa o cálculo de gradientes
    for imagem, label in batch_teste:
        saida = modelo(imagem)
        previsoes = torch.argmax(saida, dim=1)
        acertos += (previsoes == label).sum().item()
        total += label.size(0)

    # testa uma imagem real
    imagem = PIL.Image.open("numero_8.jpg").convert("L") # L = escala de cinza
    imagem = PIL.ImageOps.invert(imagem) # transforma em branco e preto
    imagem = imagem.resize((28,28))
    imagem.show()
    imagem = transform(imagem) # normaliza a imagem
    imagem = imagem.unsqueeze(0) # adiciona uma dimensão (batch) a imagem, a rede espera um tensor no formato (batch_size, quantidade_valores)

    hipotese = modelo(imagem)
    hipotese = torch.argmax(hipotese, dim=1)
    print(f'\nO numero na imagem é {hipotese.item()}')

print(f"\nAcurácia na época {epoca+1}: {(acertos/total)*100:.2f}%")