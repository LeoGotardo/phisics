import sys
import argparse
from src.controller.controller import Controller
from src.utils.dataGenerator import DataGenerator
from src.config import Config

def parseArguments():
    """
    Processa os argumentos da linha de comando.
    
    Exemplos de uso:
        python -m src.main --populate 160
        python -m src.main --populate 160 --clear
        python -m src.main --export dataset.csv
        python -m src.main --stats
    """
    parser = argparse.ArgumentParser(
        description='Talent Scout - Sistema de Análise de Atletas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python -m src.main                          # Inicia o servidor web
  python -m src.main --populate 200           # Popula DB com 200 atletas
  python -m src.main --populate 160 --clear   # Limpa DB e adiciona 160 atletas
  python -m src.main --export dados.csv       # Exporta dados para CSV
  python -m src.main --stats                  # Mostra estatísticas do DB
  python -m src.main --generate-data 100      # Gera dados sem adicionar ao DB
        """
    )
    
    # Argumentos principais
    parser.add_argument(
        '--populate',
        type=int,
        metavar='N',
        help='Popula o banco de dados com N atletas sintéticos'
    )
    
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Limpa o banco de dados antes de popular (use com --populate)'
    )
    
    parser.add_argument(
        '--export',
        type=str,
        metavar='FILE',
        help='Exporta dados do banco para arquivo CSV'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Exibe estatísticas sobre os dados no banco'
    )
    
    parser.add_argument(
        '--generate-data',
        type=int,
        metavar='N',
        help='Gera N atletas sintéticos e salva em CSV (não adiciona ao DB)'
    )
    
    return parser.parse_args()


def clearDatabase():
    """
    Limpa o banco de dados.
    """
    print("\n" + "="*70)
    print("🗑  LIMPANDO BANCO DE DADOS")
    print("="*70)
    
    from src.model.model import Model
    
    model = Model()
    
    status, message = model.clearDatabase()
    
    if status:
        print("\n✅ SUCESSO!")
        print(message)
        print("\n" + "="*70)
        return True
    else:
        print("\n❌ ERRO!")
        print(message)
        print("\n" + "="*70)
        return False


def populateDatabase(nAthletes: int, clearExisting: bool = False):
    """
    Popula o banco de dados com atletas sintéticos.
    
    Args:
        nAthletes: Número de atletas a gerar
        clearExisting: Se deve limpar dados existentes
    """
    print("\n" + "="*70)
    print("🏋️  POPULANDO BANCO DE DADOS")
    print("="*70)
    
    generator = DataGenerator(nAthletes=nAthletes)
    
    print(f"\n📊 Configuração:")
    print(f"   • Atletas a gerar: {nAthletes}")
    print(f"   • Limpar existentes: {'Sim' if clearExisting else 'Não'}")
    
    print(f"\n⏳ Gerando {nAthletes} atletas...")
    
    status, message = generator.saveToDatabase(clearExisting=clearExisting)
    
    if status:
        print("\n✅ SUCESSO!")
        print(message)
        print("\n" + "="*70)
        return True
    else:
        print("\n❌ ERRO!")
        print(message)
        print("\n" + "="*70)
        return False


def exportDatabaseToCsv(filepath: str):
    """
    Exporta dados do banco para arquivo CSV.
    
    Args:
        filepath: Caminho do arquivo de destino
    """
    print("\n" + "="*70)
    print("📤 EXPORTANDO DADOS")
    print("="*70)
    
    from src.model.model import Model
    
    model = Model()
    
    print(f"\n📂 Arquivo de destino: {filepath}")
    print(f"⏳ Exportando dados...")
    
    status, result = model.exportData()
    
    if status == True:
        with open(filepath, 'wb') as f:
            f.write(result)
        
        print(f"\n✅ Dados exportados com sucesso!")
        print(f"   Arquivo: {filepath}")
        print("\n" + "="*70)
        return True
    else:
        print(f"\n❌ Erro ao exportar: {result}")
        print("\n" + "="*70)
        return False


def showDatabaseStats():
    """
    Exibe estatísticas sobre os dados no banco.
    """
    print("\n" + "="*70)
    print("📊 ESTATÍSTICAS DO BANCO DE DADOS")
    print("="*70)
    
    from src.model.model import Model
    from src.model.athleteModel import Athlete
    
    model = Model()
    
    with Config.app.app_context():
        # Total de atletas
        totalAthletes = Config.session.query(Athlete).count()
        
        print(f"\n📈 Total de atletas: {totalAthletes}")
        
        if totalAthletes == 0:
            print("\n⚠️  Banco de dados vazio!")
            print("   Use --populate para adicionar dados")
            print("\n" + "="*70)
            return
        
        # Distribuição por cluster
        print(f"\n🎯 Distribuição por cluster:")
        
        clusterNames = {
            0: 'Iniciante',
            1: 'Intermediário',
            2: 'Competitivo',
            3: 'Elite'
        }
        
        for clusterId, clusterName in clusterNames.items():
            count = Config.session.query(Athlete).filter(
                Athlete.cluster == clusterId
            ).count()
            
            percentage = (count / totalAthletes * 100) if totalAthletes > 0 else 0
            
            bar = '█' * int(percentage / 2)
            print(f"   {clusterName:14} [{count:3}] {bar} {percentage:.1f}%")
        
        # Distribuição por sexo
        print(f"\n⚧️  Distribuição por sexo:")
        
        masculino = Config.session.query(Athlete).filter(
            Athlete.sexo == 'M'
        ).count()
        
        feminino = Config.session.query(Athlete).filter(
            Athlete.sexo == 'F'
        ).count()
        
        percMasc = (masculino / totalAthletes * 100) if totalAthletes > 0 else 0
        percFem = (feminino / totalAthletes * 100) if totalAthletes > 0 else 0
        
        print(f"   Masculino:  [{masculino:3}] {percMasc:.1f}%")
        print(f"   Feminino:   [{feminino:3}] {percFem:.1f}%")
        
        # Estatísticas de métricas
        print(f"\n📏 Médias das métricas:")
        
        import pandas as pd
        athletes = Config.session.query(Athlete).all()
        athletesData = [athlete.dict() for athlete in athletes]
        df = pd.DataFrame(athletesData)
        
        metricas = ['altura', 'envergadura', 'arremesso', 'saltoHorizontal', 'abdominais']
        
        for metrica in metricas:
            media = df[metrica].mean()
            desvio = df[metrica].std()
            minVal = df[metrica].min()
            maxVal = df[metrica].max()
            
            print(f"   {metrica:16} μ={media:6.1f} σ={desvio:5.1f} [{minVal:6.1f}, {maxVal:6.1f}]")
    
    print("\n" + "="*70)


def generateDataOnly(nAthletes: int):
    """
    Gera dados sintéticos e salva em CSV (não adiciona ao DB).
    
    Args:
        nAthletes: Número de atletas a gerar
    """
    print("\n" + "="*70)
    print("🎲 GERANDO DADOS SINTÉTICOS")
    print("="*70)
    
    generator = DataGenerator(nAthletes=nAthletes)
    
    filepath = f'dataset_athletes_{nAthletes}.csv'
    
    print(f"\n📊 Gerando {nAthletes} atletas...")
    print(f"📂 Arquivo de destino: {filepath}")
    
    status, message = generator.saveToCSV(filepath)
    
    if status:
        print("\n✅ SUCESSO!")
        print(message)
    else:
        print("\n❌ ERRO!")
        print(message)
    
    print("\n" + "="*70)


def main():
    """
    Função principal que processa argumentos e executa ações.
    """
    args = parseArguments()
    
    # Flag para controlar se o servidor deve ser iniciado
    shouldStartServer = True
    
    # Processar comandos
    if args.populate:
        success = populateDatabase(args.populate, clearExisting=args.clear)
        if not success:
            sys.exit(1)
        
        # Se --no-server, não inicia servidor
        if args.no_server:
            shouldStartServer = False
            print("\n✓ Finalizando sem iniciar servidor (--no-server)")
    
    elif args.clear:
        success = clearDatabase()
        if not success:
            sys.exit(1)
    
    elif args.export:
        success = exportDatabaseToCsv(args.export)
        if not success:
            sys.exit(1)
        
        if args.no_server:
            shouldStartServer = False
            print("\n✓ Finalizando sem iniciar servidor (--no-server)")
    
    elif args.stats:
        showDatabaseStats()
        
        if args.no_server:
            shouldStartServer = False
            print("\n✓ Finalizando sem iniciar servidor (--no-server)")
    

    # Iniciar servidor se necessário
    if shouldStartServer:
        print("\n" + "="*70)
        print("🚀 INICIANDO SERVIDOR WEB")
        print("="*70)
        print(f"\n📍 Host: {Config.HOST}")
        print(f"🔌 Porta: {Config.PORT}")
        print(f"🐛 Debug: {Config.DEBUG}")
        print(f"\n🌐 Acesse: http://{Config.HOST}:{Config.PORT}")
        print("\n💡 Pressione Ctrl+C para parar o servidor")
        print("\n" + "="*70 + "\n")
        
        Controller(Config.app)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⚠️  SERVIDOR INTERROMPIDO PELO USUÁRIO")
        print("="*70)
        sys.exit(0)
    except Exception as e:
        print("\n\n" + "="*70)
        print("❌ ERRO FATAL")
        print("="*70)
        print(f"\n{type(e).__name__}: {e}")
        
        import traceback
        print("\nTraceback:")
        traceback.print_exc()
        
        print("\n" + "="*70)
        sys.exit(1)