#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_ALCUBIERRE_RESONANCE_HUNTER.py - Zeta/Alcubierre resonance hunter with file loading
Author: Jefferson M. Okushigue
Date: 2025-08-09
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
import math

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration
mp.dps = 50  # High precision

# Constantes físicas fundamentais para Alcubierre
C_LIGHT = 299792458  # m/s
G_GRAVITY = 6.67430e-11  # m³⋅kg⁻¹⋅s⁻²
H_PLANCK = 6.62607015e-34  # J⋅s
H_BAR = H_PLANCK / (2 * math.pi)
ALPHA = 7.2973525693e-3  # Estrutura fina

# Parâmetros característicos da métrica de Alcubierre
# Velocidades de distorção superluminais
V_WARP_2C = 2 * C_LIGHT       # 5.996e8 m/s
V_WARP_5C = 5 * C_LIGHT       # 1.499e9 m/s
V_WARP_10C = 10 * C_LIGHT     # 2.998e9 m/s
V_WARP_100C = 100 * C_LIGHT   # 2.998e10 m/s

# Raios de bolha característicos
R_PLANCK = 1.616e-35    # m
R_ATOMIC = 1e-10        # m
R_NUCLEAR = 1e-15       # m
R_EARTH = 6.371e6       # m
R_SOLAR = 6.96e8        # m

# Densidades de energia exótica
RHO_PLANCK = C_LIGHT**4 / (8 * math.pi * G_GRAVITY * H_PLANCK)  # 7.267e75 kg/m³
RHO_NUCLEAR = 2.3e17    # kg/m³
RHO_VACUUM = 9.3e-27    # kg/m³

# Fatores geométricos da métrica
SIGMA_FACTOR = math.tanh(1)                    # 0.7616
THETA_FACTOR = (1/math.cosh(1))**2            # 0.4199 (sech²)
F_SHAPE = 1 / (1 + math.exp(1))               # 0.2689

# Parâmetros de eficiência Alcubierre
ETA_OPTIMAL = 1/(4*math.pi)                   # 0.0796
BETA_EXPANSION = math.sqrt(2/math.pi)         # 0.7979
EPSILON_CAUSALITY = 1/(2*math.pi)             # 0.1592

# Escalas de energia e frequência
E_WARP_PLANCK = H_PLANCK * C_LIGHT / R_PLANCK # Energia escala Planck
F_WARP_PLANCK = C_LIGHT / R_PLANCK            # Frequência de Planck

# Constantes principais para análise - valores na escala dos zeros
ALCUBIERRE_CONSTANT = V_WARP_10C / 1e8  # ~30 (escala compatível)
WARP_GEOMETRY = SIGMA_FACTOR * 10       # ~7.6
EXOTIC_DENSITY = RHO_NUCLEAR / 1e15     # ~230

TOLERANCE_LEVELS = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]  # Tolerâncias relativas
DEFAULT_TOLERANCE = 1e-8
INCREMENT = 1000  # Batch size for processing

# Constantes de controle para validação estatística
CONTROL_CONSTANTS = {
    'alcubierre_vel': V_WARP_10C / 1e8,      # ~30
    'warp_geometry': SIGMA_FACTOR * 10,       # ~7.6
    'exotic_density': RHO_NUCLEAR / 1e15,    # ~230
    'warp_efficiency': ETA_OPTIMAL * 100,     # ~7.96
    'expansion_factor': BETA_EXPANSION * 10,  # ~7.98
    'causality_limit': EPSILON_CAUSALITY * 100, # ~15.92
    'planck_ratio': E_WARP_PLANCK / 1e8,     # Escala Planck normalizada
    'random_1': 25.5,
    'random_2': 31.8,
    'random_3': 7.2,
    'random_4': 150.0,
}

MAX_WORKERS = os.cpu_count()
CACHE_FILE = "zeta_zeros_cache.pkl"
STATS_FILE = "zvt_alcubierre_stats.txt"
RESULTS_DIR = "zvt_alcubierre_results"
ZEROS_FILE = os.path.expanduser("~/Downloads/zero.txt")  # Path to the zeros file

# Parameters for analysis
FRESH_START_ZEROS = 5000
MINIMUM_FOR_STATS = 1000

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Global variable for shutdown control
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n⏸️ Shutdown solicitado. Completando lote atual e salvando...")
    shutdown_requested = True

def load_zeros_from_file(filename):
    zeros = []
    try:
        print(f"📂 Carregando zeros do arquivo: {filename}")
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                if shutdown_requested:
                    break
                line = line.strip()
                if line:
                    try:
                        zero = float(line)
                        zeros.append((line_num, zero))  # Format: (index, value)
                    except ValueError:
                        print(f"⚠️ Linha inválida {line_num}: '{line}'")
                        continue
        print(f"✅ {len(zeros):,} zeros carregados")
        return zeros
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return []

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

def load_enhanced_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list) and len(data) > 0:
                    print(f"✅ Cache válido: {len(data):,} zeros carregados")
                    return data
        except:
            print("⚠️ Cache inválido, carregando do arquivo...")
    zeros = load_zeros_from_file(ZEROS_FILE)
    if zeros:
        save_enhanced_cache(zeros)
        return zeros[:FRESH_START_ZEROS] if FRESH_START_ZEROS > 0 else zeros
    return []

def find_multi_tolerance_resonances(zeros, constants_dict=None):
    if constants_dict is None:
        constants_dict = {'alcubierre_vel': ALCUBIERRE_CONSTANT}
    all_results = {}
    for const_name, const_value in constants_dict.items():
        all_results[const_name] = {}
        for tolerance in TOLERANCE_LEVELS:
            resonances = []
            for n, gamma in zeros:
                mod_val = gamma % const_value
                min_distance = min(mod_val, const_value - mod_val)
                relative_error = min_distance / const_value  # Erro relativo
                if relative_error < tolerance:
                    resonances.append((n, gamma, min_distance, tolerance, relative_error))
            all_results[const_name][tolerance] = resonances
    return all_results

def enhanced_statistical_analysis(zeros, resonances, constant_value, tolerance):
    if len(zeros) == 0 or len(resonances) == 0:
        return None
    total_zeros = len(zeros)
    resonant_count = len(resonances)
    expected_random = total_zeros * (2 * tolerance)  # Tolerância relativa
    results = {
        'basic_stats': {
            'total_zeros': total_zeros,
            'resonant_count': resonant_count,
            'expected_random': expected_random,
            'resonance_rate': resonant_count / total_zeros,
            'significance_factor': resonant_count / expected_random if expected_random > 0 else float('inf')
        }
    }
    if expected_random >= 5:
        chi2_stat = (resonant_count - expected_random)**2 / expected_random
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=1)
        results['chi2_test'] = {
            'statistic': chi2_stat,
            'p_value': chi2_pvalue,
            'critical_value_05': 3.841,
            'significant': chi2_stat > 3.841
        }
    p_expected = 2 * tolerance  # Tolerância já é relativa
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
    results['binomial_test'] = {
        'p_value': binom_pvalue,
        'significant': binom_pvalue < 0.05
    }
    poisson_pvalue = 1 - stats.poisson.cdf(resonant_count - 1, expected_random)
    results['poisson_test'] = {
        'p_value': poisson_pvalue,
        'significant': poisson_pvalue < 0.05
    }
    return results

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

def analyze_batch_enhanced(zeros, batch_num):
    print(f"\n🔬 LOTE #{batch_num}: {len(zeros):,} zeros")
    if len(zeros) < MINIMUM_FOR_STATS:
        print(f"   📊 Necessário {MINIMUM_FOR_STATS - len(zeros):,} mais zeros para estatísticas")
        alcubierre_results = find_multi_tolerance_resonances(zeros, {'alcubierre_vel': ALCUBIERRE_CONSTANT})['alcubierre_vel']
        tolerance_summary = {}
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in alcubierre_results:
                resonances = alcubierre_results[tolerance]
                count = len(resonances)
                rate = count / len(zeros) * 100 if len(zeros) > 0 else 0
                tolerance_summary[tolerance] = {
                    'count': count,
                    'rate': rate,
                    'best_quality': min(r[4] for r in resonances) if count > 0 else None,  # Erro relativo
                    'resonances': resonances,
                    'stats': None
                }
        print(f"\n📊 RESSONÂNCIAS BÁSICAS (ALCUBIERRE):")
        print("| Tolerância | Contagem | Taxa (%) |")
        print("|------------|----------|----------|")
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in tolerance_summary:
                data = tolerance_summary[tolerance]
                print(f"| {tolerance:8.0e} | {data['count']:8d} | {data['rate']:8.3f} |")
        return tolerance_summary, {}, None
    
    alcubierre_results = find_multi_tolerance_resonances(zeros, {'alcubierre_vel': ALCUBIERRE_CONSTANT})['alcubierre_vel']
    print(f"\n📊 RESSONÂNCIAS ALCUBIERRE POR TOLERÂNCIA (Velocidade Warp = {ALCUBIERRE_CONSTANT:.3f}):")
    print("| Tolerância | Contagem | Taxa (%) | Melhor Qualidade | Significância |")
    print("|------------|----------|----------|------------------|---------------|")
    tolerance_summary = {}
    for tolerance in TOLERANCE_LEVELS:
        if tolerance in alcubierre_results:
            resonances = alcubierre_results[tolerance]
            count = len(resonances)
            rate = count / len(zeros) * 100
            if count > 0:
                best_quality = min(r[4] for r in resonances)  # Erro relativo
                stats_result = enhanced_statistical_analysis(zeros, resonances, ALCUBIERRE_CONSTANT, tolerance)
                sig_factor = stats_result['basic_stats']['significance_factor'] if stats_result else 0
                tolerance_summary[tolerance] = {
                    'count': count,
                    'rate': rate,
                    'best_quality': best_quality,
                    'significance': sig_factor,
                    'resonances': resonances,
                    'stats': stats_result
                }
                print(f"| {tolerance:8.0e} | {count:8d} | {rate:8.3f} | {best_quality:.3e} | {sig_factor:8.2f}x |")
            else:
                tolerance_summary[tolerance] = {
                    'count': 0, 'rate': 0, 'best_quality': None,
                    'significance': 0, 'resonances': [], 'stats': None
                }
                print(f"| {tolerance:8.0e} | {count:8d} | {rate:8.3f} |      N/A     |     N/A    |")
    
    print(f"\n🎛️ COMPARAÇÃO DE PARÂMETROS ALCUBIERRE (Tolerância: {DEFAULT_TOLERANCE}):")
    comparative_results = comparative_constant_analysis(zeros, DEFAULT_TOLERANCE)
    print("| Parâmetro     | Valor          | Contagem | Taxa (%) | Significância |")
    print("|---------------|----------------|----------|----------|---------------|")
    for const_name, results in comparative_results.items():
        count = results['resonance_count']
        rate = results['resonance_rate']
        sig_factor = results['stats']['basic_stats']['significance_factor'] if results['stats'] else 0
        value = results['constant_value']
        value_str = f"{value:.3f}"
        print(f"| {const_name:13s} | {value_str:14s} | {count:8d} | {rate:8.3f} | {sig_factor:8.2f}x |")
    
    best_overall = None
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['resonances']:
        best_overall = min(tolerance_summary[DEFAULT_TOLERANCE]['resonances'], key=lambda x: x[4])  # Por erro relativo
        print(f"\n💎 MELHOR RESSONÂNCIA ALCUBIERRE (Tolerância: {DEFAULT_TOLERANCE}):")
        print(f"   Zero #{best_overall[0]:,} → γ = {best_overall[1]:.15f}")
        print(f"   γ mod (v_warp) = {best_overall[2]:.15e}")
        print(f"   Erro relativo: {best_overall[4]:.15e} ({best_overall[4]*100:.12f}%)")
        print(f"   Fator de distorção: γ = {best_overall[1]/ALCUBIERRE_CONSTANT:.6f} × v_warp")
        print(f"   Velocidade superluminal: v = {best_overall[1]/ALCUBIERRE_CONSTANT:.3e} × 10c")
        print(f"   Energia exótica requerida: ρ ∝ γ²")
    
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['stats']:
        stats = tolerance_summary[DEFAULT_TOLERANCE]['stats']
        print(f"\n📈 TESTES ESTATÍSTICOS ALCUBIERRE (Tolerância: {DEFAULT_TOLERANCE}):")
        if 'chi2_test' in stats:
            chi2 = stats['chi2_test']
            print(f"   Qui-quadrado: χ² = {chi2['statistic']:.3f}, p = {chi2['p_value']:.6f}, Significativo: {'Sim' if chi2['significant'] else 'Não'}")
        if 'binomial_test' in stats:
            binom = stats['binomial_test']
            print(f"   Binomial: p = {binom['p_value']:.6f}, Significativo: {'Sim' if binom['significant'] else 'Não'}")
    
    return tolerance_summary, comparative_results, best_overall

def generate_comprehensive_report(zeros, session_results, final_batch):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(RESULTS_DIR, f"Relatorio_Alcubierre_{timestamp}.txt")
    final_tolerance_analysis, final_comparative, final_best = analyze_batch_enhanced(zeros, final_batch)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ZVT ALCUBIERRE RESONANCE HUNTER - RELATÓRIO COMPREENSIVO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {datetime.now().isoformat()}\n")
        f.write(f"Versão: Alcubierre Hunter v1.0\n")
        f.write(f"Lote Final: #{final_batch}\n")
        f.write(f"Zeros Analisados: {len(zeros):,}\n")
        f.write(f"Arquivo de Origem: {ZEROS_FILE}\n\n")
        f.write("CONFIGURAÇÃO ALCUBIERRE:\n")
        f.write(f"Velocidade da Luz (c): {C_LIGHT:.15e} m/s\n")
        f.write(f"Velocidade Warp 10c: {V_WARP_10C:.15e} m/s\n")
        f.write(f"Densidade de Planck: {RHO_PLANCK:.15e} kg/m³\n")
        f.write(f"Densidade Nuclear: {RHO_NUCLEAR:.15e} kg/m³\n")
        f.write(f"Raio de Planck: {R_PLANCK:.15e} m\n")
        f.write(f"Fator Geométrico σ: {SIGMA_FACTOR:.15f}\n")
        f.write(f"Eficiência Ótima η: {ETA_OPTIMAL:.15f}\n")
        f.write(f"Tolerâncias: {TOLERANCE_LEVELS}\n")
        f.write(f"Tolerância Padrão: {DEFAULT_TOLERANCE}\n")
        f.write(f"Precisão: {mp.dps} casas decimais\n\n")
        f.write("ANÁLISE MULTI-TOLERÂNCIA (ALCUBIERRE):\n")
        f.write("| Tolerância | Ressonâncias | Taxa (%) | Melhor Qualidade | Significância |\n")
        f.write("|------------|--------------|----------|------------------|---------------|\n")
        for tolerance in TOLERANCE_LEVELS:
            if tolerance in final_tolerance_analysis:
                data = final_tolerance_analysis[tolerance]
                best_quality = "N/A" if data['best_quality'] is None else f"{data['best_quality']:.3e}"
                significance = "N/A" if data['significance'] == 0 else f"{data['significance']:8.2f}"
                f.write(f"| {tolerance:8.0e} | {data['count']:10d} | {data['rate']:8.3f} | {best_quality:>11s} | {significance:>8s}x |\n")
        
        f.write(f"\nCOMPARAÇÃO DE PARÂMETROS ALCUBIERRE (Tolerância: {DEFAULT_TOLERANCE}):\n")
        f.write("| Parâmetro     | Valor              | Ressonâncias | Taxa (%) | Significância |\n")
        f.write("|---------------|--------------------|--------------|----------|---------------|\n")
        for const_name, results in final_comparative.items():
            value_str = f"{results['constant_value']:.6f}"
            sig_factor = results['stats']['basic_stats']['significance_factor'] if results['stats'] else 0
            f.write(f"| {const_name:13s} | {value_str:18s} | {results['resonance_count']:10d} | {results['resonance_rate']:8.3f} | {sig_factor:8.2f}x |\n")
        
        if final_best:
            f.write(f"\nMELHOR RESSONÂNCIA ALCUBIERRE:\n")
            f.write(f"Índice do Zero: #{final_best[0]:,}\n")
            f.write(f"Valor Gamma: {final_best[1]:.20f}\n")
            f.write(f"Qualidade Absoluta: {final_best[2]:.20e}\n")
            f.write(f"Qualidade Relativa: {final_best[4]:.20e}\n")
            f.write(f"Erro Relativo: {final_best[4]*100:.12f}%\n")
            f.write(f"Fator de Distorção: {final_best[1]/ALCUBIERRE_CONSTANT:.6e}\n")
            f.write(f"Velocidade Equivalente: {final_best[1]/ALCUBIERRE_CONSTANT:.3e} × 10c\n")
            f.write(f"Relação Exótica: γ = {final_best[1]/ALCUBIERRE_CONSTANT:.3e} × v_warp\n\n")
        else:
            f.write(f"\nNENHUMA RESSONÂNCIA ALCUBIERRE ENCONTRADA nas tolerâncias testadas.\n\n")
        
        if DEFAULT_TOLERANCE in final_tolerance_analysis and final_tolerance_analysis[DEFAULT_TOLERANCE]['stats']:
            stats = final_tolerance_analysis[DEFAULT_TOLERANCE]['stats']
            f.write("RESUMO ESTATÍSTICO:\n")
            basic = stats['basic_stats']
            f.write(f"  Zeros Totais: {basic['total_zeros']:,}\n")
            f.write(f"  Zeros Ressonantes: {basic['resonant_count']:,}\n")
            f.write(f"  Esperado por Acaso: {basic['expected_random']:.1f}\n")
            f.write(f"  Taxa de Ressonância: {basic['resonance_rate']*100:.6f}%\n")
            f.write(f"  Significância: {basic['significance_factor']:.3f}x\n")
            if 'chi2_test' in stats:
                f.write(f"  Qui-quadrado: χ² = {stats['chi2_test']['statistic']:.3f}, p = {stats['chi2_test']['p_value']:.6f}\n")
            if 'binomial_test' in stats:
                f.write(f"  Binomial: p = {stats['binomial_test']['p_value']:.6f}\n")
        
        f.write("\nIMPLICAÇÕES PARA FÍSICA EXÓTICA:\n")
        f.write("1. Investigar conexões entre zeros de Riemann e propulsão superluminal\n")
        f.write("2. Analisar relações com geometria de distorção espacial\n")
        f.write("3. Considerar implicações para engenharia de matéria exótica\n")
        f.write("4. Explorar conexões com estrutura fundamental do espaço-tempo\n")
        f.write("5. Avaliar viabilidade de drives de distorção baseados em zeros\n")
        f.write("="*80 + "\n")
    
    print(f"📊 Relatório Alcubierre salvo: {report_file}")
    return report_file

def infinite_hunter_alcubierre():
    global shutdown_requested
    print(f"🚀 ZVT ALCUBIERRE RESONANCE HUNTER")
    print(f"🌌 Velocidade Warp 10c = {V_WARP_10C:.3e} m/s")
    print(f"⚡ Parâmetro Principal = {ALCUBIERRE_CONSTANT:.3f}")
    print(f"📁 Arquivo: {ZEROS_FILE}")
    print(f"📊 Tolerâncias: {TOLERANCE_LEVELS}")
    print("🛑 Ctrl+C para parar")
    print("=" * 80)
    
    all_zeros = load_enhanced_cache()
    current_count = len(all_zeros)
    if current_count == 0:
        print("❌ Nenhum zero carregado. Verifique o arquivo.")
        return [], [], None
    
    print(f"📊 Zeros disponíveis: {current_count:,}")
    session_results = []
    best_overall = None
    batch_num = 1
    
    for i in range(0, current_count, INCREMENT):
        if shutdown_requested:
            break
        batch_start = i
        batch_end = min(i + INCREMENT, current_count)
        batch = all_zeros[batch_start:batch_end]
        print(f"\n🔬 LOTE #{batch_num}: Zeros {batch_start:,} a {batch_end:,}")
        start_time = time.time()
        tolerance_analysis, comparative_analysis, batch_best = analyze_batch_enhanced(batch, batch_num)
        if batch_best and (not best_overall or batch_best[4] < best_overall[4]):  # Comparar por erro relativo
            best_overall = batch_best
            print(f"    🎯 NOVO MELHOR GLOBAL ALCUBIERRE!")
            print(f"    Zero #{best_overall[0]:,} → γ mod v_warp = {best_overall[2]:.15e}")
        session_results.append({
            'batch': batch_num,
            'timestamp': datetime.now().isoformat(),
            'zeros_analyzed': batch_end,
            'batch_time': time.time() - start_time,
            'tolerance_analysis': tolerance_analysis,
            'comparative_analysis': comparative_analysis,
            'best_resonance': batch_best
        })
        batch_num += 1
    
    print(f"\n📊 Gerando relatório final Alcubierre...")
    generate_comprehensive_report(all_zeros, session_results, batch_num-1)
    return all_zeros, session_results, best_overall

def main():
    """Função principal"""
    global shutdown_requested
    print("🌟 Iniciando ZVT Alcubierre Resonance Hunter")
    
    try:
        zeros, results, best = infinite_hunter_alcubierre()
        
        if zeros and len(zeros) > 0:
            print(f"\n🎯 Análise Alcubierre Concluída!")
            print(f"📁 Resultados em: {RESULTS_DIR}/")
            print(f"💾 Cache: {CACHE_FILE}")
            print(f"📊 Estatísticas: {STATS_FILE}")
            
            if best:
                print(f"🏆 Melhor ressonância: Zero #{best[0]:,} com qualidade relativa {best[4]:.3e}")
                print(f"🚀 Fator Warp: γ = {best[1]/ALCUBIERRE_CONSTANT:.3e} × v_warp")
                print(f"⚡ Velocidade superluminal: {best[1]/ALCUBIERRE_CONSTANT:.1f} × 10c")
            else:
                print(f"📊 Nenhuma ressonância Alcubierre excepcional encontrada")
                print(f"🤔 Zeros de Riemann podem ser independentes da métrica de distorção")
                
    except KeyboardInterrupt:
        print(f"\n⏹️ Análise interrompida. Progresso salvo.")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n🔬 Sessão Alcubierre concluída!")

if __name__ == "__main__":
    shutdown_requested = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
