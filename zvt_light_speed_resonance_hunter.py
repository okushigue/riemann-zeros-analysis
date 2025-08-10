#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_LIGHT_SPEED_RESONANCE_HUNTER.py - Zeta/Light Speed resonance hunter with file loading
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

# Velocidade da luz e constantes fundamentais relacionadas
C_LIGHT = 299792458  # m/s (valor exato por definição)
H_PLANCK = 6.62607015e-34  # J⋅s
H_BAR = H_PLANCK / (2 * math.pi)
E_ELECTRON = 1.602176634e-19  # C
M_ELECTRON = 9.1093837015e-31  # kg
M_PROTON = 1.67262192369e-27  # kg
ALPHA = 7.2973525693e-3  # Estrutura fina
EPSILON_0 = 8.8541878128e-12  # F/m (permissividade do vácuo)

# Quantidades derivadas da velocidade da luz
# Energia de massa do elétron: E = mc²
E_ELECTRON_REST = M_ELECTRON * C_LIGHT**2  # ≈ 8.19e-14 J ≈ 0.511 MeV

# Energia de massa do próton: E = mc²
E_PROTON_REST = M_PROTON * C_LIGHT**2  # ≈ 1.50e-10 J ≈ 938 MeV

# Comprimento de onda Compton do elétron: λ_C = h/(mc)
LAMBDA_COMPTON_E = H_PLANCK / (M_ELECTRON * C_LIGHT)  # ≈ 2.43e-12 m

# Comprimento de onda Compton do próton
LAMBDA_COMPTON_P = H_PLANCK / (M_PROTON * C_LIGHT)  # ≈ 1.32e-15 m

# Raio clássico do elétron: r_e = e²/(4πε₀mc²)
R_CLASSICAL_E = E_ELECTRON**2 / (4 * math.pi * EPSILON_0 * M_ELECTRON * C_LIGHT**2)  # ≈ 2.82e-15 m

# Constante de estrutura fina em termos de c: α = e²/(4πε₀ℏc)
ALPHA_HC = ALPHA * H_BAR * C_LIGHT  # ≈ 2.31e-30 J⋅m

# Escalas normalizadas para compatibilidade com zeros de Riemann
C_SCALED = C_LIGHT / 1e8           # ≈ 3.0 (velocidade normalizada)
C2_SCALED = C_LIGHT**2 / 1e16      # ≈ 9.0 (c² normalizada)
HC_SCALED = H_PLANCK * C_LIGHT / 1e26  # ≈ 2.0 (ℏc normalizada)
MC2_E_SCALED = E_ELECTRON_REST * 1e13  # ≈ 8.19 (energia elétron)
MC2_P_SCALED = E_PROTON_REST * 1e10    # ≈ 15.0 (energia próton)

# Fatores relativísticos característicos
GAMMA_1_5 = 1 / math.sqrt(1 - 0.5**2)      # γ para v = 0.5c ≈ 1.15
GAMMA_0_9 = 1 / math.sqrt(1 - 0.9**2)      # γ para v = 0.9c ≈ 2.29
GAMMA_0_99 = 1 / math.sqrt(1 - 0.99**2)    # γ para v = 0.99c ≈ 7.09

# Escalas de tempo relativísticas
TIME_LIGHT_METER = 1 / C_LIGHT * 1e9       # Tempo para luz viajar 1m (ns) ≈ 3.34
TIME_LIGHT_KM = 1000 / C_LIGHT * 1e6       # Tempo para luz viajar 1km (μs) ≈ 3.34

# Constante principal para análise
LIGHT_SPEED_CONSTANT = C_SCALED  # ≈ 3.0

TOLERANCE_LEVELS = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]  # Tolerâncias relativas
DEFAULT_TOLERANCE = 1e-8
INCREMENT = 1000  # Batch size for processing

# Constantes de controle para validação estatística
CONTROL_CONSTANTS = {
    'light_speed': C_SCALED,              # ≈ 3.0
    'light_squared': C2_SCALED,           # ≈ 9.0
    'planck_light': HC_SCALED,            # ≈ 2.0
    'electron_energy': MC2_E_SCALED,      # ≈ 8.19
    'proton_energy': MC2_P_SCALED,        # ≈ 15.0
    'gamma_half_c': GAMMA_1_5,            # ≈ 1.15
    'gamma_0_9c': GAMMA_0_9,              # ≈ 2.29
    'gamma_0_99c': GAMMA_0_99,            # ≈ 7.09
    'time_meter_ns': TIME_LIGHT_METER,    # ≈ 3.34
    'alpha_scaled': ALPHA * 1000,         # ≈ 7.30
    'random_1': 2.8,
    'random_2': 3.2,
    'random_3': 8.5,
    'random_4': 15.7,
}

MAX_WORKERS = os.cpu_count()
CACHE_FILE = "zeta_zeros_cache.pkl"
STATS_FILE = "zvt_light_speed_stats.txt"
RESULTS_DIR = "zvt_light_speed_results"
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
        constants_dict = {'light_speed': LIGHT_SPEED_CONSTANT}
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
        light_results = find_multi_tolerance_resonances(zeros, {'light_speed': LIGHT_SPEED_CONSTANT})['light_speed']
        tolerance_summary = {}
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in light_results:
                resonances = light_results[tolerance]
                count = len(resonances)
                rate = count / len(zeros) * 100 if len(zeros) > 0 else 0
                tolerance_summary[tolerance] = {
                    'count': count,
                    'rate': rate,
                    'best_quality': min(r[4] for r in resonances) if count > 0 else None,  # Erro relativo
                    'resonances': resonances,
                    'stats': None
                }
        print(f"\n📊 RESSONÂNCIAS BÁSICAS (VELOCIDADE DA LUZ):")
        print("| Tolerância | Contagem | Taxa (%) |")
        print("|------------|----------|----------|")
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in tolerance_summary:
                data = tolerance_summary[tolerance]
                print(f"| {tolerance:8.0e} | {data['count']:8d} | {data['rate']:8.3f} |")
        return tolerance_summary, {}, None
    
    light_results = find_multi_tolerance_resonances(zeros, {'light_speed': LIGHT_SPEED_CONSTANT})['light_speed']
    print(f"\n📊 RESSONÂNCIAS VELOCIDADE DA LUZ POR TOLERÂNCIA (c/10⁸ = {LIGHT_SPEED_CONSTANT:.3f}):")
    print("| Tolerância | Contagem | Taxa (%) | Melhor Qualidade | Significância |")
    print("|------------|----------|----------|------------------|---------------|")
    tolerance_summary = {}
    for tolerance in TOLERANCE_LEVELS:
        if tolerance in light_results:
            resonances = light_results[tolerance]
            count = len(resonances)
            rate = count / len(zeros) * 100
            if count > 0:
                best_quality = min(r[4] for r in resonances)  # Erro relativo
                stats_result = enhanced_statistical_analysis(zeros, resonances, LIGHT_SPEED_CONSTANT, tolerance)
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
    
    print(f"\n🎛️ COMPARAÇÃO DE PARÂMETROS RELATIVÍSTICOS (Tolerância: {DEFAULT_TOLERANCE}):")
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
        print(f"\n💎 MELHOR RESSONÂNCIA VELOCIDADE DA LUZ (Tolerância: {DEFAULT_TOLERANCE}):")
        print(f"   Zero #{best_overall[0]:,} → γ = {best_overall[1]:.15f}")
        print(f"   γ mod (c/10⁸) = {best_overall[2]:.15e}")
        print(f"   Erro relativo: {best_overall[4]:.15e} ({best_overall[4]*100:.12f}%)")
        print(f"   Fator relativístico: γ = {best_overall[1]/LIGHT_SPEED_CONSTANT:.6f} × (c/10⁸)")
        print(f"   Velocidade correspondente: v = {best_overall[1]/LIGHT_SPEED_CONSTANT:.3e} × 10⁸ m/s")
        print(f"   Fração da velocidade da luz: v/c = {(best_overall[1]/LIGHT_SPEED_CONSTANT)/3:.6f}")
        print(f"   Energia relativística (elétron): E = γm_e c² = {(best_overall[1]/LIGHT_SPEED_CONSTANT) * 0.511:.3f} MeV")
        print(f"   Tempo de luz (1 metro): t = {(best_overall[1]/LIGHT_SPEED_CONSTANT) * 3.34:.3f} ns")
    
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['stats']:
        stats = tolerance_summary[DEFAULT_TOLERANCE]['stats']
        print(f"\n📈 TESTES ESTATÍSTICOS RELATIVÍSTICOS (Tolerância: {DEFAULT_TOLERANCE}):")
        if 'chi2_test' in stats:
            chi2 = stats['chi2_test']
            print(f"   Qui-quadrado: χ² = {chi2['statistic']:.3f}, p = {chi2['p_value']:.6f}, Significativo: {'Sim' if chi2['significant'] else 'Não'}")
        if 'binomial_test' in stats:
            binom = stats['binomial_test']
            print(f"   Binomial: p = {binom['p_value']:.6f}, Significativo: {'Sim' if binom['significant'] else 'Não'}")
    
    return tolerance_summary, comparative_results, best_overall

def generate_comprehensive_report(zeros, session_results, final_batch):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(RESULTS_DIR, f"Relatorio_Light_Speed_{timestamp}.txt")
    final_tolerance_analysis, final_comparative, final_best = analyze_batch_enhanced(zeros, final_batch)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ZVT LIGHT SPEED RESONANCE HUNTER - RELATÓRIO COMPREENSIVO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {datetime.now().isoformat()}\n")
        f.write(f"Versão: Light Speed Hunter v1.0\n")
        f.write(f"Lote Final: #{final_batch}\n")
        f.write(f"Zeros Analisados: {len(zeros):,}\n")
        f.write(f"Arquivo de Origem: {ZEROS_FILE}\n\n")
        f.write("CONFIGURAÇÃO RELATIVÍSTICA:\n")
        f.write(f"Velocidade da Luz (c): {C_LIGHT:,} m/s\n")
        f.write(f"Energia de Repouso do Elétron: {E_ELECTRON_REST:.15e} J ({E_ELECTRON_REST/E_ELECTRON:.6f} eV)\n")
        f.write(f"Energia de Repouso do Próton: {E_PROTON_REST:.15e} J ({E_PROTON_REST/E_ELECTRON:.6f} eV)\n")
        f.write(f"Comprimento de Onda Compton (elétron): {LAMBDA_COMPTON_E:.15e} m\n")
        f.write(f"Comprimento de Onda Compton (próton): {LAMBDA_COMPTON_P:.15e} m\n")
        f.write(f"Raio Clássico do Elétron: {R_CLASSICAL_E:.15e} m\n")
        f.write(f"Fator Lorentz (v=0.5c): {GAMMA_1_5:.15f}\n")
        f.write(f"Fator Lorentz (v=0.9c): {GAMMA_0_9:.15f}\n")
        f.write(f"Fator Lorentz (v=0.99c): {GAMMA_0_99:.15f}\n")
        f.write(f"Tolerâncias: {TOLERANCE_LEVELS}\n")
        f.write(f"Tolerância Padrão: {DEFAULT_TOLERANCE}\n")
        f.write(f"Precisão: {mp.dps} casas decimais\n\n")
        f.write("ANÁLISE MULTI-TOLERÂNCIA (VELOCIDADE DA LUZ):\n")
        f.write("| Tolerância | Ressonâncias | Taxa (%) | Melhor Qualidade | Significância |\n")
        f.write("|------------|--------------|----------|------------------|---------------|\n")
        for tolerance in TOLERANCE_LEVELS:
            if tolerance in final_tolerance_analysis:
                data = final_tolerance_analysis[tolerance]
                best_quality = "N/A" if data['best_quality'] is None else f"{data['best_quality']:.3e}"
                significance = "N/A" if data['significance'] == 0 else f"{data['significance']:8.2f}"
                f.write(f"| {tolerance:8.0e} | {data['count']:10d} | {data['rate']:8.3f} | {best_quality:>11s} | {significance:>8s}x |\n")
        
        f.write(f"\nCOMPARAÇÃO DE PARÂMETROS RELATIVÍSTICOS (Tolerância: {DEFAULT_TOLERANCE}):\n")
        f.write("| Parâmetro     | Valor              | Ressonâncias | Taxa (%) | Significância |\n")
        f.write("|---------------|--------------------|--------------|----------|---------------|\n")
        for const_name, results in final_comparative.items():
            value_str = f"{results['constant_value']:.6f}"
            sig_factor = results['stats']['basic_stats']['significance_factor'] if results['stats'] else 0
            f.write(f"| {const_name:13s} | {value_str:18s} | {results['resonance_count']:10d} | {results['resonance_rate']:8.3f} | {sig_factor:8.2f}x |\n")
        
        if final_best:
            f.write(f"\nMELHOR RESSONÂNCIA VELOCIDADE DA LUZ:\n")
            f.write(f"Índice do Zero: #{final_best[0]:,}\n")
            f.write(f"Valor Gamma: {final_best[1]:.20f}\n")
            f.write(f"Qualidade Absoluta: {final_best[2]:.20e}\n")
            f.write(f"Qualidade Relativa: {final_best[4]:.20e}\n")
            f.write(f"Erro Relativo: {final_best[4]*100:.12f}%\n")
            f.write(f"Fator Relativístico: {final_best[1]/LIGHT_SPEED_CONSTANT:.6e}\n")
            f.write(f"Velocidade Correspondente: {final_best[1]/LIGHT_SPEED_CONSTANT:.3e} × 10⁸ m/s\n")
            f.write(f"Fração de c: {(final_best[1]/LIGHT_SPEED_CONSTANT)/3:.6f}\n")
            f.write(f"Energia Relativística (e⁻): {(final_best[1]/LIGHT_SPEED_CONSTANT) * 0.511:.6f} MeV\n")
            f.write(f"Relação Relativística: γ = {final_best[1]/LIGHT_SPEED_CONSTANT:.3e} × (c/10⁸)\n\n")
        else:
            f.write(f"\nNENHUMA RESSONÂNCIA VELOCIDADE DA LUZ ENCONTRADA nas tolerâncias testadas.\n\n")
        
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
        
        f.write("\nIMPLICAÇÕES PARA FÍSICA RELATIVÍSTICA:\n")
        f.write("1. Investigar conexões entre zeros de Riemann e relatividade especial\n")
        f.write("2. Analisar relações com invariância de Lorentz\n")
        f.write("3. Considerar implicações para estrutura do espaço-tempo\n")
        f.write("4. Explorar conexões com causalidade e velocidade limite\n")
        f.write("5. Avaliar potencial para teorias unificadas baseadas em zeros\n")
        f.write("="*80 + "\n")
    
    print(f"📊 Relatório Velocidade da Luz salvo: {report_file}")
    return report_file

def infinite_hunter_light_speed():
    global shutdown_requested
    print(f"🚀 ZVT LIGHT SPEED RESONANCE HUNTER")
    print(f"⚡ Velocidade da Luz c = {C_LIGHT:,} m/s")
    print(f"🔬 Parâmetro Principal = {LIGHT_SPEED_CONSTANT:.3f}")
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
            print(f"    🎯 NOVO MELHOR GLOBAL VELOCIDADE DA LUZ!")
            print(f"    Zero #{best_overall[0]:,} → γ mod (c/10⁸) = {best_overall[2]:.15e}")
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
    
    print(f"\n📊 Gerando relatório final Velocidade da Luz...")
    generate_comprehensive_report(all_zeros, session_results, batch_num-1)
    return all_zeros, session_results, best_overall

def main():
    """Função principal"""
    global shutdown_requested
    print("🌟 Iniciando ZVT Light Speed Resonance Hunter")
    
    try:
        zeros, results, best = infinite_hunter_light_speed()
        
        if zeros and len(zeros) > 0:
            print(f"\n🎯 Análise Velocidade da Luz Concluída!")
            print(f"📁 Resultados em: {RESULTS_DIR}/")
            print(f"💾 Cache: {CACHE_FILE}")
            print(f"📊 Estatísticas: {STATS_FILE}")
            
            if best:
                print(f"🏆 Melhor ressonância: Zero #{best[0]:,} com qualidade relativa {best[4]:.3e}")
                print(f"⚡ Fator relativístico: γ = {best[1]/LIGHT_SPEED_CONSTANT:.3e} × (c/10⁸)")
                print(f"🔬 Velocidade: v = {(best[1]/LIGHT_SPEED_CONSTANT)/3:.6f}c")
                print(f"💫 Energia (elétron): E = {(best[1]/LIGHT_SPEED_CONSTANT) * 0.511:.3f} MeV")
            else:
                print(f"📊 Nenhuma ressonância Velocidade da Luz excepcional encontrada")
                print(f"🤔 Zeros de Riemann podem ser independentes da relatividade especial")
                
    except KeyboardInterrupt:
        print(f"\n⏹️ Análise interrompida. Progresso salvo.")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n🔬 Sessão Velocidade da Luz concluída!")

if __name__ == "__main__":
    shutdown_requested = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
