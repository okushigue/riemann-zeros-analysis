#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_FOUR_FORCES_HUNTER.py - Zeta/4 Forças Fundamentais resonance hunter
Author: Jefferson M. Okushigue
Date: 2025-08-10
Modificado para buscar ressonâncias com as 4 forças fundamentais
"""

import numpy as np
from mpmath import mp
import time
from concurrent.futures import ProcessPoolExecutor
import pickle
import os
import signal
import sys
from datetime import datetime
from scipy import stats
from scipy.stats import kstest, anderson
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration
mp.dps = 50  # High precision

# 4 FORÇAS FUNDAMENTAIS DA FÍSICA
FUNDAMENTAL_FORCES = {
    'eletromagnetica': 1 / 137.035999084,      # α (constante de estrutura fina)
    'forte': 0.1185,                           # αs (constante de acoplamento forte at MZ)
    'fraca': 0.0338,                           # αW (constante de acoplamento fraca)
    'gravitacional': 5.906e-39                 # αG (constante gravitacional adimensional)
}

# Tolerâncias específicas para cada força (ajustadas às suas magnitudes)
FORCE_TOLERANCES = {
    'eletromagnetica': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    'forte': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
    'fraca': [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    'gravitacional': [1e-38, 1e-39, 1e-40, 1e-41, 1e-42, 1e-43]  # Tolerâncias apropriadas para escala gravitacional
}

# Constantes de controle para validação estatística
CONTROL_CONSTANTS = {
    'eletromagnetica': FUNDAMENTAL_FORCES['eletromagnetica'],
    'forte': FUNDAMENTAL_FORCES['forte'],
    'fraca': FUNDAMENTAL_FORCES['fraca'], 
    'random_1': 1 / 142.7,
    'random_2': 1 / 129.3,
    'golden_ratio': (np.sqrt(5) - 1) / 2,
    'pi_scale': np.pi / 100,
    'e_scale': np.e / 100
}

TOLERANCE_LEVELS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
DEFAULT_TOLERANCE = 1e-5
INCREMENT = 10000  # Lotes maiores para processar mais rapidamente os 2M+ zeros

# CRITÉRIOS PARA RESSONÂNCIA SIGNIFICATIVA
SIGNIFICANCE_CRITERIA = {
    'min_resonances': 10,           # Mínimo de ressonâncias
    'min_significance_factor': 2.0, # Fator de significância mínimo (2x o esperado)
    'max_p_value': 0.01,           # p-value máximo para significância
    'min_chi2_stat': 6.635         # χ² crítico para p < 0.01
}

MAX_WORKERS = os.cpu_count()
CACHE_FILE = "zeta_zeros_cache.pkl"
STATS_FILE = "zvt_4forces_stats.txt"
RESULTS_DIR = "zvt_4forces_results"
ZEROS_FILE = os.path.expanduser("~/zeta/zero.txt")  # Path to the zeros file

# Parameters for analysis
FRESH_START_ZEROS = 0  # 0 = usar todos os zeros disponíveis
MINIMUM_FOR_STATS = 1000

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Global variable for shutdown control
shutdown_requested = False

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n⏸️ Shutdown solicitado. Completando lote atual e salvando...")
    shutdown_requested = True

# Função para verificar se uma ressonância é significativa
def is_significant_resonance(resonances, stats_result, force_name, constant_value, tolerance):
    """Verifica se uma ressonância atende aos critérios de significância"""
    if not stats_result or len(resonances) < SIGNIFICANCE_CRITERIA['min_resonances']:
        return False, "Insuficientes ressonâncias"
    
    basic_stats = stats_result['basic_stats']
    significance_factor = basic_stats['significance_factor']
    
    # Verificar fator de significância
    if significance_factor < SIGNIFICANCE_CRITERIA['min_significance_factor']:
        return False, f"Significância baixa: {significance_factor:.2f}x"
    
    # Verificar testes estatísticos
    significant_tests = []
    
    if 'chi2_test' in stats_result:
        chi2 = stats_result['chi2_test']
        if chi2['statistic'] > SIGNIFICANCE_CRITERIA['min_chi2_stat']:
            significant_tests.append(f"Chi2: χ²={chi2['statistic']:.3f}")
    
    if 'binomial_test' in stats_result:
        binom = stats_result['binomial_test']
        if binom['p_value'] < SIGNIFICANCE_CRITERIA['max_p_value']:
            significant_tests.append(f"Binomial: p={binom['p_value']:.2e}")
    
    if 'poisson_test' in stats_result:
        poisson = stats_result['poisson_test']
        if poisson['p_value'] < SIGNIFICANCE_CRITERIA['max_p_value']:
            significant_tests.append(f"Poisson: p={poisson['p_value']:.2e}")
    
    if len(significant_tests) > 0:
        return True, f"Testes significativos: {', '.join(significant_tests)}"
    
    return False, "Nenhum teste estatístico significativo"

# Função para log de ressonância significativa (sem pausa)
def log_significant_resonance(force_name, constant_value, tolerance, resonances, stats_result, zeros_count):
    """Registra ressonância significativa no log sem pausar"""
    print("\n" + "🚨" * 40)
    print("🚨 RESSONÂNCIA SIGNIFICATIVA DETECTADA! 🚨")
    print("🚨" * 40)
    print(f"🔬 FORÇA: {force_name.upper()}")
    print(f"🎯 CONSTANTE: {constant_value:.15e}")
    print(f"📏 TOLERÂNCIA: {tolerance:.0e}")
    print(f"📊 ZEROS ANALISADOS: {zeros_count:,}")
    print(f"🔍 RESSONÂNCIAS: {len(resonances):,}")
    
    if stats_result:
        basic = stats_result['basic_stats']
        print(f"📈 Taxa: {basic['resonance_rate']*100:.6f}% | Significância: {basic['significance_factor']:.3f}x")
        
        if 'chi2_test' in stats_result:
            chi2 = stats_result['chi2_test']
            print(f"🧪 χ²={chi2['statistic']:.3f}, p={chi2['p_value']:.2e}")
    
    # Mostrar as 5 melhores ressonâncias
    if resonances:
        best_resonances = sorted(resonances, key=lambda x: x[2])[:5]
        print(f"\n💎 TOP 5 MELHORES RESSONÂNCIAS:")
        print("| Rank | Zero #    | Gamma            | Qualidade        | Energia (GeV) |")
        print("|------|-----------|------------------|------------------|---------------|")
        for i, (n, gamma, quality, _) in enumerate(best_resonances, 1):
            energy = gamma / 10  # Estimativa de energia
            print(f"| {i:4d} | {n:9,} | {gamma:16.12f} | {quality:.6e} | {energy:13.3f} |")
    
    print("🚨" * 40)
    print("▶️ CONTINUANDO ANÁLISE AUTOMATICAMENTE...\n")

# Load zeros from a file with progress indicator
def load_zeros_from_file(filename):
    zeros = []
    try:
        print(f"📂 Carregando zeros do arquivo: {filename}")
        
        # Primeiro, contar total de linhas para mostrar progresso
        print("📊 Contando linhas do arquivo...")
        with open(filename, 'r') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"📊 Total de linhas encontradas: {total_lines:,}")
        
        with open(filename, 'r') as f:
            progress_counter = 0
            for line_num, line in enumerate(f, start=1):
                if shutdown_requested:
                    break
                line = line.strip()
                if line:
                    try:
                        zero = float(line)
                        zeros.append((line_num, zero))  # Format: (index, value)
                        progress_counter += 1
                        
                        # Mostrar progresso a cada 100,000 zeros
                        if progress_counter % 100000 == 0:
                            percent = (progress_counter / total_lines) * 100
                            print(f"📈 Carregados {progress_counter:,} zeros ({percent:.1f}%)")
                            
                    except ValueError:
                        print(f"⚠️ Linha inválida {line_num}: '{line}'")
                        continue
        print(f"✅ {len(zeros):,} zeros carregados com sucesso de {total_lines:,} linhas")
        return zeros
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return []

# Save zeros to cache
def save_enhanced_cache(zeros, backup=True):
    try:
        if backup and os.path.exists(CACHE_FILE):
            backup_file = f"{CACHE_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(CACHE_FILE, backup_file)
            print(f"📦 Backup do cache criado: {backup_file}")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(zeros, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"💾 Cache salvo: {len(zeros)} zeros")
    except Exception as e:
        print(f"❌ Erro ao salvar cache: {e}")

# Load zeros from cache or file with forced reload option
def load_enhanced_cache(force_reload=False):
    if not force_reload and os.path.exists(CACHE_FILE):
        try:
            print(f"🔍 Verificando cache existente...")
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list) and len(data) > 0:
                    print(f"✅ Cache válido: {len(data):,} zeros carregados")
                    
                    # Verificar se o cache parece completo
                    file_size = os.path.getsize(ZEROS_FILE)
                    expected_zeros = file_size // 20  # Estimativa aproximada (20 bytes por zero)
                    
                    if len(data) < expected_zeros * 0.5:  # Se cache tem menos de 50% do esperado
                        print(f"⚠️ Cache parece incompleto ({len(data):,} vs ~{expected_zeros:,} esperados)")
                        print(f"🔄 Forçando recarga do arquivo...")
                        force_reload = True
                    else:
                        return data
        except Exception as e:
            print(f"⚠️ Cache inválido ({e}), carregando do arquivo...")
            force_reload = True
    
    if force_reload or not os.path.exists(CACHE_FILE):
        print("📂 Carregando todos os zeros do arquivo original...")
        zeros = load_zeros_from_file(ZEROS_FILE)
        if zeros:
            print(f"💾 Salvando {len(zeros):,} zeros no cache...")
            save_enhanced_cache(zeros)
            return zeros
    return []

# Find resonances at multiple tolerance levels with force-specific tolerances
def find_multi_tolerance_resonances(zeros, constants_dict=None):
    if constants_dict is None:
        constants_dict = FUNDAMENTAL_FORCES
    all_results = {}
    for const_name, const_value in constants_dict.items():
        all_results[const_name] = {}
        
        # Usar tolerâncias específicas para cada força ou tolerâncias genéricas para outras constantes
        if const_name in FORCE_TOLERANCES:
            tolerances_to_use = FORCE_TOLERANCES[const_name]
        else:
            tolerances_to_use = TOLERANCE_LEVELS
            
        for tolerance in tolerances_to_use:
            resonances = []
            for n, gamma in zeros:
                mod_val = gamma % const_value
                min_distance = min(mod_val, const_value - mod_val)
                if min_distance < tolerance:
                    resonances.append((n, gamma, min_distance, tolerance))
            all_results[const_name][tolerance] = resonances
    return all_results

# Enhanced statistical analysis with validation
def enhanced_statistical_analysis(zeros, resonances, constant_value, tolerance):
    if len(zeros) == 0 or len(resonances) == 0:
        return None
    total_zeros = len(zeros)
    resonant_count = len(resonances)
    expected_random = total_zeros * (2 * tolerance / constant_value)
    
    # Validação básica para evitar valores inválidos
    if constant_value <= 0 or tolerance <= 0:
        return None
        
    results = {
        'basic_stats': {
            'total_zeros': total_zeros,
            'resonant_count': resonant_count,
            'expected_random': expected_random,
            'resonance_rate': resonant_count / total_zeros,
            'significance_factor': resonant_count / expected_random if expected_random > 0 else float('inf')
        }
    }
    
    # Teste qui-quadrado (apenas se esperado >= 5)
    if expected_random >= 5:
        chi2_stat = (resonant_count - expected_random)**2 / expected_random
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=1)
        results['chi2_test'] = {
            'statistic': chi2_stat,
            'p_value': chi2_pvalue,
            'critical_value_05': 3.841,
            'significant': chi2_stat > 3.841
        }
    
    # Teste binomial com validação de probabilidade
    p_expected = 2 * tolerance / constant_value
    
    # Verificar se p_expected está no intervalo válido [0,1]
    if 0 <= p_expected <= 1:
        try:
            binom_result = stats.binomtest(resonant_count, total_zeros, p_expected, alternative='two-sided')
            binom_pvalue = binom_result.pvalue
        except AttributeError:
            try:
                binom_pvalue = stats.binom_test(resonant_count, total_zeros, p_expected, alternative='two-sided')
            except AttributeError:
                from scipy.stats import binom
                binom_pvalue = 2 * min(binom.cdf(resonant_count, total_zeros, p_expected),
                                      1 - binom.cdf(resonant_count - 1, total_zeros, p_expected))
        except Exception as e:
            print(f"    Aviso: Teste binomial falhou: {e}")
            binom_pvalue = 1.0  # p-value neutro em caso de erro
            
        results['binomial_test'] = {
            'p_value': binom_pvalue,
            'significant': binom_pvalue < 0.05
        }
    else:
        # Se p_expected é inválido, não fazer teste binomial
        print(f"    Aviso: p_expected={p_expected:.2e} fora do intervalo [0,1] - pulando teste binomial")
        results['binomial_test'] = {
            'p_value': 1.0,  # p-value neutro
            'significant': False,
            'invalid_probability': True
        }
    
    # Teste de Poisson (mais robusto para casos extremos)
    if expected_random > 0:
        try:
            poisson_pvalue = 1 - stats.poisson.cdf(resonant_count - 1, expected_random)
            results['poisson_test'] = {
                'p_value': poisson_pvalue,
                'significant': poisson_pvalue < 0.05
            }
        except Exception as e:
            print(f"    Aviso: Teste de Poisson falhou: {e}")
            results['poisson_test'] = {
                'p_value': 1.0,
                'significant': False
            }
    
    return results

# Compare resonances between different constants
def comparative_constant_analysis(zeros, tolerance=DEFAULT_TOLERANCE):
    multi_results = find_multi_tolerance_resonances(zeros, CONTROL_CONSTANTS)
    comparative_stats = {}
    for const_name, const_results in multi_results.items():
        if tolerance in const_results:
            resonances = const_results[tolerance]
            const_value = CONTROL_CONSTANTS[const_name]
            stats_result = enhanced_statistical_analysis(zeros, resonances, const_value, tolerance)
            comparative_stats[const_name] = {
                'constant_value': const_value,
                'resonance_count': len(resonances),
                'resonance_rate': len(resonances) / len(zeros) * 100,
                'stats': stats_result
            }
    return comparative_stats

# Analyze a batch of zeros with significance detection (sem pausa)
def analyze_batch_with_significance_detection(zeros, batch_num):
    print(f"\n🔬 LOTE #{batch_num}: {len(zeros):,} zeros")
    
    # Análise das 4 forças fundamentais
    forces_results = find_multi_tolerance_resonances(zeros, FUNDAMENTAL_FORCES)
    
    significant_found = False
    best_resonances = {}  # Armazenar melhores ressonâncias por força
    
    print(f"\n🌌 ANÁLISE DAS 4 FORÇAS FUNDAMENTAIS:")
    print("| Força         | Constante    | Tolerância | Ressonâncias | Taxa (%) | Significância |")
    print("|---------------|--------------|------------|--------------|----------|---------------|")
    
    for force_name, force_value in FUNDAMENTAL_FORCES.items():
        # Usar tolerâncias específicas para cada força
        tolerances_to_check = FORCE_TOLERANCES.get(force_name, TOLERANCE_LEVELS[:3])
        
        for tolerance in tolerances_to_check[:3]:  # Verificar apenas as 3 principais tolerâncias de cada força
            try:
                if tolerance in forces_results[force_name]:
                    resonances = forces_results[force_name][tolerance]
                    count = len(resonances)
                    rate = count / len(zeros) * 100 if len(zeros) > 0 else 0
                    
                    stats_result = enhanced_statistical_analysis(zeros, resonances, force_value, tolerance)
                    sig_factor = stats_result['basic_stats']['significance_factor'] if stats_result else 0
                    
                    # Mostrar status de significância na tabela
                    sig_marker = "🚨" if sig_factor > SIGNIFICANCE_CRITERIA['min_significance_factor'] else "  "
                    print(f"|{sig_marker}{force_name:11s} | {force_value:.12e} | {tolerance:8.0e} | {count:10d} | {rate:8.3f} | {sig_factor:8.2f}x |")
                    
                    # Guardar melhor ressonância para cada força
                    if resonances:
                        current_best = min(resonances, key=lambda x: x[2])
                        if force_name not in best_resonances or current_best[2] < best_resonances[force_name][2]:
                            best_resonances[force_name] = current_best
                    
                    # Verificar se é significativo e registrar no log
                    is_sig, reason = is_significant_resonance(resonances, stats_result, force_name, force_value, tolerance)
                    if is_sig:
                        significant_found = True
                        log_significant_resonance(force_name, force_value, tolerance, resonances, stats_result, len(zeros))
                else:
                    print(f"|  {force_name:11s} | {force_value:.12e} | {tolerance:8.0e} |        N/A |      N/A |      N/A |")
            except Exception as e:
                print(f"|❌{force_name:11s} | {force_value:.12e} | {tolerance:8.0e} |      ERRO |      N/A |      N/A |")
                print(f"    Erro na análise de {force_name}: {e}")
                continue
    
    # Mostrar melhores ressonâncias encontradas para cada força
    if best_resonances:
        print(f"\n💎 MELHORES RESSONÂNCIAS ENCONTRADAS:")
        print("| Força         | Zero #    | Gamma            | Qualidade      | Energia (GeV) |")
        print("|---------------|-----------|------------------|----------------|---------------|")
        for force_name, (n, gamma, quality, tolerance) in best_resonances.items():
            energy = gamma / 10  # Estimativa de energia
            print(f"| {force_name:13s} | {n:9,} | {gamma:16.12f} | {quality:.6e} | {energy:13.3f} |")
    
    # Análise comparativa sempre continua (com tratamento de erro)
    try:
        comparative_analysis = comparative_constant_analysis(zeros)
    except Exception as e:
        print(f"⚠️ Erro na análise comparativa: {e}")
        comparative_analysis = {}
    
    return forces_results, comparative_analysis, best_resonances, 'continue'

# Generate a comprehensive report
def generate_comprehensive_report(zeros, session_results, final_batch):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(RESULTS_DIR, f"Relatorio_4Forcas_{timestamp}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ZVT 4 FORÇAS FUNDAMENTAIS HUNTER - RELATÓRIO COMPREENSIVO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {datetime.now().isoformat()}\n")
        f.write(f"Versão: Four Forces Hunter v1.0\n")
        f.write(f"Lote Final: #{final_batch}\n")
        f.write(f"Zeros Analisados: {len(zeros):,}\n")
        f.write(f"Arquivo de Origem: {ZEROS_FILE}\n\n")
        
        f.write("CONFIGURAÇÃO DAS 4 FORÇAS FUNDAMENTAIS:\n")
        for force_name, force_value in FUNDAMENTAL_FORCES.items():
            tolerances_str = ", ".join([f"{t:.0e}" for t in FORCE_TOLERANCES.get(force_name, TOLERANCE_LEVELS)])
            f.write(f"  {force_name.capitalize()}: {force_value:.15e} (tolerâncias: {tolerances_str})\n")
        
        f.write(f"\nPrecisão: {mp.dps} casas decimais\n\n")
        
        f.write("CRITÉRIOS DE SIGNIFICÂNCIA:\n")
        for key, value in SIGNIFICANCE_CRITERIA.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"📊 Relatório salvo: {report_file}")
    return report_file

# Main function
def infinite_hunter_four_forces():
    global shutdown_requested
    print(f"🚀 ZVT 4 FORÇAS FUNDAMENTAIS HUNTER")
    print(f"🌌 Buscando ressonâncias com as 4 forças fundamentais da física")
    print("=" * 80)
    
    for force_name, force_value in FUNDAMENTAL_FORCES.items():
        tolerances_str = ", ".join([f"{t:.0e}" for t in FORCE_TOLERANCES.get(force_name, TOLERANCE_LEVELS)[:3]])
        print(f"🔬 {force_name.upper()}: {force_value:.15e} (tolerâncias: {tolerances_str})")
    
    print(f"\n📁 Arquivo: {ZEROS_FILE}")
    print("🛑 Ctrl+C para parar")
    print("🚨 Ressonâncias significativas serão destacadas automaticamente")
    print("=" * 80)
    
    # Verificar se deve forçar recarga
    force_reload = False
    if len(sys.argv) > 1 and sys.argv[1] == '--force-reload':
        force_reload = True
        print("🔄 MODO FORÇA RECARGA ATIVADO - Recarregando do arquivo original")
    
    all_zeros = load_enhanced_cache(force_reload=force_reload)
    current_count = len(all_zeros)
    
    if current_count == 0:
        print("❌ Nenhum zero carregado. Verifique o arquivo.")
        return [], [], None
    
    if current_count == 0:
        print("❌ Nenhum zero carregado. Verifique o arquivo.")
        return [], [], None
    
    # Diagnóstico do arquivo original
    try:
        file_size = os.path.getsize(ZEROS_FILE)
        print(f"📊 Arquivo original: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        with open(ZEROS_FILE, 'r') as f:
            total_lines = sum(1 for line in f if line.strip())
        print(f"📊 Linhas no arquivo: {total_lines:,}")
        print(f"📊 Zeros carregados: {current_count:,}")
        
        if current_count < total_lines * 0.9:  # Se carregou menos de 90% das linhas
            print(f"⚠️ AVISO: Carregou apenas {(current_count/total_lines)*100:.1f}% do arquivo!")
            print(f"💡 Execute com: python3 {sys.argv[0]} --force-reload")
            
    except Exception as e:
        print(f"⚠️ Erro ao verificar arquivo: {e}")
    
    print(f"🎯 PROCESSANDO TODOS OS {current_count:,} ZEROS!")
    print(f"📦 Lotes de {INCREMENT:,} zeros cada")
    print(f"⏱️ Estimativa: ~{(current_count//INCREMENT)} lotes")
    
    session_results = []
    best_overall = {}  # Melhores ressonâncias globais por força
    batch_num = 1
    
    for i in range(0, current_count, INCREMENT):
        if shutdown_requested:
            break
        
        batch_start = i
        batch_end = min(i + INCREMENT, current_count)
        batch = all_zeros[batch_start:batch_end]
        
        # Indicador de progresso
        progress_percent = (batch_end / current_count) * 100
        print(f"\n🔬 LOTE #{batch_num}: Zeros {batch_start:,} a {batch_end:,} ({progress_percent:.1f}% concluído)")
        start_time = time.time()
        
        forces_analysis, comparative_analysis, batch_best, decision = analyze_batch_with_significance_detection(batch, batch_num)
        
        # Atualizar melhores ressonâncias globais
        if batch_best:
            for force_name, (n, gamma, quality, tolerance) in batch_best.items():
                if force_name not in best_overall or quality < best_overall[force_name][2]:
                    best_overall[force_name] = (n, gamma, quality, tolerance)
                    print(f"    🎯 NOVO MELHOR GLOBAL para {force_name.upper()}!")
                    print(f"    Zero #{n:,} → γ={gamma:.12f}, qualidade={quality:.6e}")
        
        elapsed = time.time() - start_time
        zeros_per_sec = len(batch) / elapsed if elapsed > 0 else 0
        remaining_zeros = current_count - batch_end
        eta_seconds = remaining_zeros / zeros_per_sec if zeros_per_sec > 0 else 0
        eta_hours = eta_seconds / 3600
        
        print(f"⏱️ Lote processado em {elapsed:.1f}s ({zeros_per_sec:,.0f} zeros/s)")
        if eta_hours > 0:
            print(f"📈 ETA para conclusão: {eta_hours:.1f} horas")
        
        session_results.append({
            'batch': batch_num,
            'timestamp': datetime.now().isoformat(),
            'zeros_analyzed': batch_end,
            'batch_time': elapsed,
            'forces_analysis': forces_analysis,
            'comparative_analysis': comparative_analysis,
            'best_resonance': batch_best,
            'progress_percent': progress_percent
        })
        
        batch_num += 1
    
    print(f"\n📊 Gerando relatório final...")
    generate_comprehensive_report(all_zeros, session_results, batch_num-1)
    
    # Mostrar melhores ressonâncias globais
    if best_overall:
        print(f"\n" + "🏆" * 60)
        print(f"🏆 MELHORES RESSONÂNCIAS GLOBAIS ENCONTRADAS 🏆")
        print(f"🏆" * 60)
        print("| Força         | Zero #        | Gamma                | Qualidade        | Energia (GeV) |")
        print("|---------------|---------------|----------------------|------------------|---------------|")
        for force_name, (n, gamma, quality, tolerance) in best_overall.items():
            energy = gamma / 10
            print(f"| {force_name:13s} | {n:13,} | {gamma:20.15f} | {quality:.6e} | {energy:13.6f} |")
        
        print(f"\n💎 DETALHES DAS MELHORES RESSONÂNCIAS:")
        for force_name, (n, gamma, quality, tolerance) in best_overall.items():
            error_percent = (quality / FUNDAMENTAL_FORCES[force_name]) * 100
            print(f"\n🔬 {force_name.upper()}:")
            print(f"   Zero #{n:,} (γ = {gamma:.15f})")
            print(f"   Qualidade: {quality:.15e}")
            print(f"   Erro relativo: {error_percent:.12f}%")
            print(f"   Tolerância: {tolerance:.0e}")
            print(f"   Energia estimada: {gamma/10:.6f} GeV")
        print(f"🏆" * 60)
    
    # Resumo final
    total_processed = session_results[-1]['zeros_analyzed'] if session_results else 0
    print(f"\n" + "="*60)
    print(f"📊 RESUMO FINAL DA ANÁLISE")
    print(f"="*60)
    print(f"📈 Total de zeros processados: {total_processed:,} de {current_count:,}")
    print(f"📈 Porcentagem concluída: {(total_processed/current_count)*100:.1f}%")
    print(f"📈 Lotes processados: {len(session_results)}")
    if session_results:
        total_time = sum(r['batch_time'] for r in session_results)
        avg_time = total_time / len(session_results)
        print(f"📈 Tempo total: {total_time:.1f}s ({total_time/3600:.1f}h)")
        print(f"📈 Tempo médio por lote: {avg_time:.1f}s")
        print(f"📈 Velocidade média: {total_processed/total_time:,.0f} zeros/s")
    print(f"="*60)
    
    return all_zeros, session_results, best_overall

# Main execution
if __name__ == "__main__":
    shutdown_requested = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("🌟 Iniciando ZVT 4 Forças Fundamentais Hunter")
        zeros, results, best = infinite_hunter_four_forces()
        
        if zeros and len(zeros) > 0:
            print(f"\n🎯 Análise Concluída!")
            print(f"📁 Resultados em: {RESULTS_DIR}/")
            print(f"💾 Cache: {CACHE_FILE}")
            print(f"📊 Estatísticas: {STATS_FILE}")
    except KeyboardInterrupt:
        print(f"\n⏹️ Análise interrompida. Progresso salvo.")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n🔬 Sessão concluída!")
