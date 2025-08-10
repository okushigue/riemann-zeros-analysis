#!/usr/bin/env python3
"""
LAUNCHER - APLICAÇÕES COMPUTACIONAIS ZEROS ZETA
Hub central para todas as aplicações usando zeros como ferramenta computacional

🧮 APLICAÇÕES DISPONÍVEIS:
1. 📊 Análise de Correlações Físicas (corrigida)
2. 🧮 Toolkit Computacional (π, γ, funções especiais)
3. 🔢 Computador de Primos (π(x), gaps, PNT)
4. 🔐 Gerador Criptográfico (chaves, PRNG)
5. 🤖 Preditor ML (próximo zero, padrões)
6. 📈 Análise Espectral Avançada
"""

import os
import sys
import subprocess
import glob
from datetime import datetime

class ZetaComputationalLauncher:
    def __init__(self):
        self.available_scripts = self._detect_scripts()
        self.zeros_file = self._detect_zeros_file()
        
        print("🧮 LAUNCHER - APLICAÇÕES COMPUTACIONAIS ZETA")
        print("="*50)
        print(f"📅 Sessão iniciada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Zeros detectados: {self.zeros_file}")
        print(f"🎯 Scripts disponíveis: {len(self.available_scripts)}")
        print()
        
        self._display_banner()
    
    def _detect_scripts(self):
        """Detecta scripts disponíveis no diretório"""
        script_patterns = {
            'physics': ['*physics*analyzer*.py', 'working_zeta_analyzer.py'],
            'toolkit': ['*computational*toolkit*.py', 'zeta_computational_toolkit.py'],
            'primes': ['*prime*computer*.py', 'zeta_prime_computer.py'],
            'crypto': ['*crypto*generator*.py', 'zeta_crypto_generator.py'],
            'ml': ['*ml*predictor*.py', 'zeta_ml_predictor.py'],
        }
        
        found_scripts = {}
        
        for category, patterns in script_patterns.items():
            for pattern in patterns:
                matches = glob.glob(pattern)
                if matches:
                    found_scripts[category] = matches[0]
                    break
        
        return found_scripts
    
    def _detect_zeros_file(self):
        """Detecta arquivo de zeros automaticamente"""
        # Padrões em ordem de prioridade
        patterns = [
            'zeta_zeros*.txt',
            'zeros*.txt',
            'riemann*.txt',
            '*zeros*.txt',
            '*.txt'
        ]
        
        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                # Retornar o maior arquivo (provavelmente tem mais zeros)
                largest_file = max(files, key=os.path.getsize)
                return largest_file
        
        return None
    
    def _display_banner(self):
        """Exibe banner informativo"""
        print("🎯 APLICAÇÕES COMPUTACIONAIS USANDO ZEROS ZETA")
        print("=" * 50)
        print("Transforme seus 2M+ zeros em ferramenta computacional!")
        print()
        
        if not self.zeros_file:
            print("⚠️  ATENÇÃO: Nenhum arquivo de zeros detectado")
            print("💡 Certifique-se de ter um arquivo .txt com zeros")
            print()
    
    def display_main_menu(self):
        """Exibe menu principal das aplicações"""
        print("🧮 APLICAÇÕES DISPONÍVEIS:")
        print("-" * 30)
        
        applications = [
            {
                'id': 'physics',
                'name': '📊 Análise de Correlações Físicas',
                'description': 'Testa correlações entre zeros e constantes físicas (metodologia corrigida)',
                'file': self.available_scripts.get('physics'),
                'icon': '🔬'
            },
            {
                'id': 'toolkit',
                'name': '🧮 Toolkit Computacional',
                'description': 'Computa π, γ, funções especiais usando zeros',
                'file': self.available_scripts.get('toolkit'),
                'icon': '⚙️'
            },
            {
                'id': 'primes',
                'name': '🔢 Computador de Primos',
                'description': 'Calcula π(x), gaps entre primos, verifica PNT',
                'file': self.available_scripts.get('primes'),
                'icon': '🎯'
            },
            {
                'id': 'crypto',
                'name': '🔐 Gerador Criptográfico',
                'description': 'Gera chaves criptográficas e números aleatórios seguros',
                'file': self.available_scripts.get('crypto'),
                'icon': '🛡️'
            },
            {
                'id': 'ml',
                'name': '🤖 Preditor Machine Learning',
                'description': 'Prediz próximo zero, detecta padrões, análise ML',
                'file': self.available_scripts.get('ml'),
                'icon': '🧠'
            },
        ]
        
        # Mostrar aplicações disponíveis
        available_apps = []
        for i, app in enumerate(applications, 1):
            if app['file'] and os.path.exists(app['file']):
                print(f"   {i}. {app['icon']} {app['name']}")
                print(f"      {app['description']}")
                print(f"      📁 {app['file']}")
                available_apps.append(app)
            else:
                print(f"   {i}. ❌ {app['name']} (não disponível)")
            print()
        
        # Opções adicionais
        print("   6. 📋 Informações do Sistema")
        print("   7. 🛠️  Instalar Scripts Faltantes") 
        print("   8. ❌ Sair")
        print()
        
        return available_apps
    
    def run_application(self, app_info):
        """Executa aplicação selecionada"""
        script_file = app_info['file']
        
        print(f"\n🚀 EXECUTANDO: {app_info['name']}")
        print("=" * 40)
        print(f"📁 Script: {script_file}")
        
        if self.zeros_file:
            print(f"📊 Zeros: {self.zeros_file}")
        
        print("⏱️  Iniciando...")
        print()
        
        try:
            # Executar script com arquivo de zeros como argumento
            if self.zeros_file:
                result = subprocess.run([sys.executable, script_file, self.zeros_file], 
                                      capture_output=False, text=True)
            else:
                result = subprocess.run([sys.executable, script_file], 
                                      capture_output=False, text=True)
            
            if result.returncode == 0:
                print(f"\n✅ {app_info['name']} executado com sucesso!")
            else:
                print(f"\n⚠️  {app_info['name']} terminou com código {result.returncode}")
                
        except FileNotFoundError:
            print(f"❌ Erro: Python não encontrado")
        except Exception as e:
            print(f"❌ Erro ao executar: {e}")
        
        input("\n📋 Pressione Enter para voltar ao menu...")
    
    def show_system_info(self):
        """Mostra informações do sistema"""
        print(f"\n📋 INFORMAÇÕES DO SISTEMA")
        print("=" * 30)
        
        # Informações do arquivo de zeros
        if self.zeros_file and os.path.exists(self.zeros_file):
            size_mb = os.path.getsize(self.zeros_file) / (1024 * 1024)
            print(f"📊 ARQUIVO DE ZEROS:")
            print(f"   📁 Nome: {self.zeros_file}")
            print(f"   💾 Tamanho: {size_mb:.1f} MB")
            
            # Tentar contar linhas
            try:
                with open(self.zeros_file, 'r') as f:
                    line_count = sum(1 for line in f if line.strip() and not line.startswith('#'))
                print(f"   📏 Linhas: {line_count:,}")
                
                # Verificar primeiras linhas
                with open(self.zeros_file, 'r') as f:
                    sample_lines = []
                    for i, line in enumerate(f):
                        if i >= 3:
                            break
                        line = line.strip()
                        if line and not line.startswith('#'):
                            sample_lines.append(line)
                
                if sample_lines:
                    print(f"   🔍 Amostra: {sample_lines}")
                    
            except Exception as e:
                print(f"   ❌ Erro ao analisar: {e}")
        else:
            print(f"📊 ARQUIVO DE ZEROS: ❌ Não encontrado")
        
        print(f"\n🎯 SCRIPTS DETECTADOS:")
        for category, script in self.available_scripts.items():
            status = "✅" if os.path.exists(script) else "❌"
            print(f"   {status} {category}: {script}")
        
        print(f"\n🐍 AMBIENTE PYTHON:")
        print(f"   Versão: {sys.version}")
        print(f"   Executável: {sys.executable}")
        
        # Verificar dependências
        try:
            import numpy
            print(f"   NumPy: ✅ {numpy.__version__}")
        except ImportError:
            print(f"   NumPy: ❌ Não instalado")
        
        try:
            import scipy
            print(f"   SciPy: ✅ {scipy.__version__}")
        except ImportError:
            print(f"   SciPy: ❌ Não instalado")
        
        input("\n📋 Pressione Enter para voltar...")
    
    def install_missing_scripts(self):
        """Ajuda a instalar scripts faltantes"""
        print(f"\n🛠️  INSTALAÇÃO DE SCRIPTS FALTANTES")
        print("=" * 35)
        
        missing_scripts = []
        for category, script in self.available_scripts.items():
            if not script or not os.path.exists(script):
                missing_scripts.append(category)
        
        if not missing_scripts:
            print("✅ Todos os scripts estão disponíveis!")
        else:
            print("❌ Scripts faltantes detectados:")
            for script in missing_scripts:
                print(f"   • {script}")
            
            print(f"\n💡 SOLUÇÕES:")
            print("1. Certifique-se de ter baixado todos os scripts")
            print("2. Verifique se os nomes dos arquivos estão corretos")
            print("3. Scripts esperados:")
            
            expected_names = {
                'physics': 'working_zeta_analyzer.py',
                'toolkit': 'zeta_computational_toolkit.py', 
                'primes': 'zeta_prime_computer.py',
                'crypto': 'zeta_crypto_generator.py',
                'ml': 'zeta_ml_predictor.py'
            }
            
            for category in missing_scripts:
                if category in expected_names:
                    print(f"   • {category}: {expected_names[category]}")
        
        input("\n📋 Pressione Enter para voltar...")
    
    def run_interactive_session(self):
        """Executa sessão interativa principal"""
        while True:
            try:
                # Limpar tela (funciona na maioria dos terminais)
                os.system('clear' if os.name == 'posix' else 'cls')
                
                # Mostrar banner
                self._display_banner()
                
                # Mostrar menu
                available_apps = self.display_main_menu()
                
                # Obter escolha do usuário
                choice = input("🎯 Escolha uma aplicação (1-8): ").strip()
                
                if choice == '8':
                    print("\n👋 Encerrando launcher...")
                    break
                elif choice == '6':
                    self.show_system_info()
                elif choice == '7':
                    self.install_missing_scripts()
                elif choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(available_apps):
                        app = available_apps[choice_num - 1]
                        self.run_application(app)
                    else:
                        print("❌ Opção inválida!")
                        input("Pressione Enter para continuar...")
                else:
                    print("❌ Entrada inválida!")
                    input("Pressione Enter para continuar...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Encerrando launcher...")
                break
            except Exception as e:
                print(f"\n❌ Erro inesperado: {e}")
                input("Pressione Enter para continuar...")

def main():
    """Função principal do launcher"""
    print("🧮 INICIALIZANDO LAUNCHER COMPUTACIONAL...")
    
    launcher = ZetaComputationalLauncher()
    
    if not launcher.zeros_file:
        print("\n⚠️  AVISO: Nenhum arquivo de zeros detectado")
        print("💡 Para melhor experiência, tenha um arquivo .txt com zeros")
        print("📁 Formatos suportados: um zero por linha, 'índice: valor', etc.")
        
        response = input("\n🔥 Continuar mesmo assim? (s/N): ").strip().lower()
        if response != 's':
            print("❌ Launcher cancelado")
            return
    
    # Executar sessão interativa
    launcher.run_interactive_session()

if __name__ == "__main__":
    main()
