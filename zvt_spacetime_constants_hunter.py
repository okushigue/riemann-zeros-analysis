#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_SPACETIME_CONSTANTS_HUNTER.py - Zeta/Spacetime Constants resonance hunter
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

# Constantes fundamentais do espaço-tempo
K_BOLTZMANN = 1.380649e-23        # J/K (CODATA 2018)
E_CHARGE = 1.602176634e-19        # C (carga elementar)
EPSILON_0 = 8.8541878128e-12      # F/m (permissividade do vácuo)
MU_0 = 4 * math.pi * 1e-7         # H/m (permeabilidade do vácuo)
M_ELECTRON = 9.1093837015e-31     # kg (massa do elétron)
M_PROTON = 1.67262192369e-27      # kg (massa do próton)
M_NEUTRON = 1.67492749804e-27     # kg (massa do nêutron)

# Constantes já descobertas (para comparação)
C_LIGHT = 299792458               # m/s
H_PLANCK = 6.62607015e-34        # J⋅s
G_GRAVITY = 6.67430e-11          # m³⋅kg⁻¹⋅s⁻²
ALPHA = 7.2973525693e-3          # Estrutura fina

# Constantes derivadas importantes
H_BAR = H_PLANCK / (2 * math.pi)
STEFAN_BOLTZMANN = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
AVOGADRO = 6.02214076e23          # mol⁻¹
R_GAS = K_BOLTZMANN * AVOGADRO    # J⋅mol⁻¹⋅K⁻¹

# Escalas normalizadas para compatibilidade com zeros de Riemann
# Normalizando para ordem de grandeza ~1-100 (compatível com zeros)

# Grupo 1: Constantes Termodinâmicas
KB_SCALED = K_BOLTZMANN * 1e23     # ≈ 1.38 (Boltzmann normalizada)
STEFAN_SCALED = STEFAN_BOLTZMANN * 1e8  # ≈ 5.67 (Stefan-Boltzmann)
R_GAS_SCALED = R_GAS / 10          # ≈ 0.831 (constante dos gases)
AVOGADRO_SCALED = AVOGADRO / 1e23  # ≈ 6.02 (número de Avogadro)

# Grupo 2: Constantes Eletromagnéticas
E_CHARGE_SCALED = E_CHARGE * 1e19  # ≈ 1.60 (carga elementar)
EPSILON0_SCALED = EPSILON_0 * 1e12 # ≈ 8.85 (permissividade)
MU0_SCALED = MU_0 * 1e7           # ≈ 12.57 (permeabilidade)
IMPEDANCE_0 = math.sqrt(MU_0/EPSILON_0) / 10  # ≈ 37.7 Ω (impedância do vácuo)

# Grupo 3: Massas Fundamentais
ME_SCALED = M_ELECTRON * 1e31     # ≈ 9.11 (massa elétron)
MP_SCALED = M_PROTON * 1e27       # ≈ 1.67 (massa próton)
MN_SCALED = M_NEUTRON * 1e27      # ≈ 1.67 (massa nêutron)
MASS_RATIO_PE = M_PROTON / M_ELECTRON  # ≈ 1836.15 (razão próton/elétron)

# Grupo 4: Energias Características
KB_TEMP_ROOM = K_BOLTZMANN * 300 * 1e21    # ≈ 4.14 (energia térmica ambiente)
ELECTRON_VOLT = E_CHARGE * 1e19             # ≈ 1.60 (1 eV em Joules normalizado)
THERMAL_VOLTAGE = K_BOLTZMANN * 300 / E_CHARGE  # ≈ 0.0259 V a 300K
COMPTON_E = H_PLANCK / (M_ELECTRON * C_LIGHT) * 1e12  # ≈ 2.43 (Compton elétron)

# Grupo 5: Frequências e Tempos
PLASMA_FREQ_E = math.sqrt(E_CHARGE**2 / (EPSILON_0 * M_ELECTRON)) / 1e10  # frequência plasma
CYCLOTRON_FREQ = E_CHARGE / M_ELECTRON / 1e10  # fator ciclotron
BOHR_FREQ = (M_ELECTRON * E_CHARGE**4) / (4 * math.pi * EPSILON_0**2 * H_BAR**3) / 1e15  # freq Bohr

# Constante principal para análise
SPACETIME_CONSTANT = KB_SCALED  # Começando com Boltzmann

TOLERANCE_LEVELS = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]  # Tolerâncias relativas
DEFAULT_TOLERANCE = 1e-8
INCREMENT = 1000  # Batch size for processing

# Constantes de controle para validação estatística
CONTROL_CONSTANTS = {
    # Grupo Termodinâmico
    'boltzmann': KB_SCALED,              # ≈ 1.38
    'stefan_boltz': STEFAN_SCALED,       # ≈ 5.67
    'gas_constant': R_GAS_SCALED,        # ≈ 0.831
    'avogadro': AVOGADRO_SCALED,         # ≈ 6.02
    'thermal_room': KB_TEMP_ROOM,        # ≈ 4.14
    
    # Grupo Eletromagnético  
    'elem_charge': E_CHARGE_SCALED,      # ≈ 1.60
    'epsilon_0': EPSILON0_SCALED,        # ≈ 8.85
    'mu_0': MU0_SCALED,                  # ≈ 12.57
    'impedance_vac': IMPEDANCE_0,        # ≈ 37.7
    'electron_volt': ELECTRON_VOLT,      # ≈ 1.60
    
    # Grupo Massas
    'mass_electron': ME_SCALED,          # ≈ 9.11
    'mass_proton': MP_SCALED,            # ≈ 1.67
    'mass_neutron': MN_SCALED,           # ≈ 1.67
    'mass_ratio_pe': MASS_RATIO_PE / 100, # ≈ 18.36
    'compton_electron': COMPTON_E,       # ≈ 2.43
    
    # Controles aleatórios
    'random_1': 1.5,
    'random_2': 5.8,
    'random_3': 12.3,
    'random_4': 37.2,
}

MAX_WORKERS = os.cpu_count()
CACHE_FILE = "zeta_zeros_cache.pkl"
STATS_FILE = "zvt_spacetime_stats.txt"
RESULTS_DIR = "zvt_spacetime_results"
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
        constants_dict = {'boltzmann': SPACETIME_CONSTANT}
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
        spacetime_results = find_multi_tolerance_resonances(zeros, {'boltzmann': SPACETIME_CONSTANT})['boltzmann']
        tolerance_summary = {}
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in spacetime_results:
                resonances = spacetime_results[tolerance]
                count = len(resonances)
                rate = count / len(zeros) * 100 if len(zeros) > 0 else 0
                tolerance_summary[tolerance] = {
                    'count': count,
                    'rate': rate,
                    'best_quality': min(r[4] for r in resonances) if count > 0 else None,  # Erro relativo
                    'resonances': resonances,
                    'stats': None
                }
        print(f"\n📊 RESSONÂNCIAS BÁSICAS (CONSTANTES ESPAÇO-TEMPO):")
        print("| Tolerância | Contagem | Taxa (%) |")
        print("|------------|----------|----------|")
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in tolerance_summary:
                data = tolerance_summary[tolerance]
                print(f"| {tolerance:8.0e} | {data['count']:8d} | {data['rate']:8.3f} |")
        return tolerance_summary, {}, None
    
    spacetime_results = find_multi_tolerance_resonances(zeros, {'boltzmann': SPACETIME_CONSTANT})['boltzmann']
    print(f"\n📊 RESSONÂNCIAS CONSTANTES ESPAÇO-TEMPO POR TOLERÂNCIA (k_B×10²³ = {SPACETIME_CONSTANT:.3f}):")
    print("| Tolerância | Contagem | Taxa (%) | Melhor Qualidade | Significância |")
    print("|------------|----------|----------|------------------|---------------|")
    tolerance_summary = {}
    for tolerance in TOLERANCE_LEVELS:
        if tolerance in spacetime_results:
            resonances = spacetime_results[tolerance]
            count = len(resonances)
            rate = count / len(zeros) * 100
            if count > 0:
                best_quality = min(r[4] for r in resonances)  # Erro relativo
                stats_result = enhanced_statistical_analysis(zeros, resonances, SPACETIME_CONSTANT, tolerance)
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
    
    print(f"\n🎛️ COMPARAÇÃO DE CONSTANTES ESPAÇO-TEMPO (Tolerância: {DEFAULT_TOLERANCE}):")
    comparative_results = comparative_constant_analysis(zeros, DEFAULT_TOLERANCE)
    print("| Constante       | Valor          | Contagem | Taxa (%) | Significância |")
    print("|-----------------|----------------|----------|----------|---------------|")
    for const_name, results in comparative_results.items():
        count = results['resonance_count']
        rate = results['resonance_rate']
        sig_factor = results['stats']['basic_stats']['significance_factor'] if results['stats'] else 0
        value = results['constant_value']
        value_str = f"{value:.3f}"
        print(f"| {const_name:15s} | {value_str:14s} | {count:8d} | {rate:8.3f} | {sig_factor:8.2f}x |")
    
    best_overall = None
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['resonances']:
        best_overall = min(tolerance_summary[DEFAULT_TOLERANCE]['resonances'], key=lambda x: x[4])  # Por erro relativo
        print(f"\n💎 MELHOR RESSONÂNCIA ESPAÇO-TEMPO (Tolerância: {DEFAULT_TOLERANCE}):")
        print(f"   Zero #{best_overall[0]:,} → γ = {best_overall[1]:.15f}")
        print(f"   γ mod (k_B×10²³) = {best_overall[2]:.15e}")
        print(f"   Erro relativo: {best_overall[4]:.15e} ({best_overall[4]*100:.12f}%)")
        print(f"   Fator termodinâmico: γ = {best_overall[1]/SPACETIME_CONSTANT:.6f} × (k_B×10²³)")
        print(f"   Temperatura equivalente: T = {best_overall[1]/(KB_SCALED*1e-23):.3e} K")
        print(f"   Energia térmica: E = k_B×T = {best_overall[1]*1e-23:.3e} J")
        print(f"   Temperatura em eV: T = {best_overall[1]/(KB_SCALED*1e-23)*K_BOLTZMANN/E_CHARGE:.6f} eV")
    
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['stats']:
        stats = tolerance_summary[DEFAULT_TOLERANCE]['stats']
        print(f"\n📈 TESTES ESTATÍSTICOS ESPAÇO-TEMPO (Tolerância: {DEFAULT_TOLERANCE}):")
        if 'chi2_test' in stats:
            chi2 = stats['chi2_test']
            print(f"   Qui-quadrado: χ² = {chi2['statistic']:.3f}, p = {chi2['p_value']:.6f}, Significativo: {'Sim' if chi2['significant'] else 'Não'}")
        if 'binomial_test' in stats:
            binom = stats['binomial_test']
            print(f"   Binomial: p = {binom['p_value']:.6f}, Significativo: {'Sim' if binom['significant'] else 'Não'}")
    
    return tolerance_summary, comparative_results, best_overall

def generate_comprehensive_report(zeros, session_results, final_batch):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(RESULTS_DIR, f"Relatorio_Spacetime_{timestamp}.txt")
    final_tolerance_analysis, final_comparative, final_best = analyze_batch_enhanced(zeros, final_batch)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ZVT SPACETIME CONSTANTS RESONANCE HUNTER - RELATÓRIO COMPREENSIVO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {datetime.now().isoformat()}\n")
        f.write(f"Versão: Spacetime Constants Hunter v1.0\n")
        f.write(f"Lote Final: #{final_batch}\n")
        f.write(f"Zeros Analisados: {len(zeros):,}\n")
        f.write(f"Arquivo de Origem: {ZEROS_FILE}\n\n")
        f.write("CONFIGURAÇÃO CONSTANTES ESPAÇO-TEMPO:\n")
        f.write(f"Constante de Boltzmann (k_B): {K_BOLTZMANN:.15e} J/K\n")
        f.write(f"Carga Elementar (e): {E_CHARGE:.15e} C\n")
        f.write(f"Permissividade do Vácuo (ε₀): {EPSILON_0:.15e} F/m\n")
        f.write(f"Permeabilidade do Vácuo (μ₀): {MU_0:.15e} H/m\n")
        f.write(f"Massa do Elétron (m_e): {M_ELECTRON:.15e} kg\n")
        f.write(f"Massa do Próton (m_p): {M_PROTON:.15e} kg\n")
        f.write(f"Impedância do Vácuo: {math.sqrt(MU_0/EPSILON_0):.15f} Ω\n")
        f.write(f"Constante de Stefan-Boltzmann: {STEFAN_BOLTZMANN:.15e} W⋅m⁻²⋅K⁻⁴\n")
        f.write(f"Número de Avogadro: {AVOGADRO:.15e} mol⁻¹\n")
        f.write(f"Tolerâncias: {TOLERANCE_LEVELS}\n")
        f.write(f"Tolerância Padrão: {DEFAULT_TOLERANCE}\n")
        f.write(f"Precisão: {mp.dps} casas decimais\n\n")
        f.write("ANÁLISE MULTI-TOLERÂNCIA (CONSTANTES ESPAÇO-TEMPO):\n")
        f.write("| Tolerância | Ressonâncias | Taxa (%) | Melhor Qualidade | Significância |\n")
        f.write("|------------|--------------|----------|------------------|---------------|\n")
        for tolerance in TOLERANCE_LEVELS:
            if tolerance in final_tolerance_analysis:
                data = final_tolerance_analysis[tolerance]
                best_quality = "N/A" if data['best_quality'] is None else f"{data['best_quality']:.3e}"
                significance = "N/A" if data['significance'] == 0 else f"{data['significance']:8.2f}"
                f.write(f"| {tolerance:8.0e} | {data['count']:10d} | {data['rate']:8.3f} | {best_quality:>11s} | {significance:>8s}x |\n")
        
        f.write(f"\nCOMPARAÇÃO DE CONSTANTES ESPAÇO-TEMPO (Tolerância: {DEFAULT_TOLERANCE}):\n")
        f.write("| Constante         | Valor              | Ressonâncias | Taxa (%) | Significância |\n")
        f.write("|-------------------|--------------------|--------------|----------|---------------|\n")
        for const_name, results in final_comparative.items():
            value_str = f"{results['constant_value']:.6f}"
            sig_factor = results['stats']['basic_stats']['significance_factor'] if results['stats'] else 0
            f.write(f"| {const_name:17s} | {value_str:18s} | {results['resonance_count']:10d} | {results['resonance_rate']:8.3f} | {sig_factor:8.2f}x |\n")
        
        if final_best:
            f.write(f"\nMELHOR RESSONÂNCIA ESPAÇO-TEMPO:\n")
            f.write(f"Índice do Zero: #{final_best[0]:,}\n")
            f.write(f"Valor Gamma: {final_best[1]:.20f}\n")
            f.write(f"Qualidade Absoluta: {final_best[2]:.20e}\n")
            f.write(f"Qualidade Relativa: {final_best[4]:.20e}\n")
            f.write(f"Erro Relativo: {final_best[4]*100:.12f}%\n")
            f.write(f"Fator Termodinâmico: {final_best[1]/SPACETIME_CONSTANT:.6e}\n")
            f.write(f"Temperatura Equivalente: {final_best[1]/(KB_SCALED*1e-23):.3e} K\n")
            f.write(f"Energia Térmica: {final_best[1]*1e-23:.3e} J\n")
            f.write(f"Relação Fundamental: γ = {final_best[1]/SPACETIME_CONSTANT:.3e} × k_B\n\n")
        else:
            f.write(f"\nNENHUMA RESSONÂNCIA ESPAÇO-TEMPO ENCONTRADA nas tolerâncias testadas.\n\n")
        
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
        
        f.write("\nIMPLICAÇÕES PARA FÍSICA FUNDAMENTAL:\n")
        f.write("1. Investigar conexões entre zeros de Riemann e termodinâmica\n")
        f.write("2. Analisar relações com eletromagnetismo fundamental\n")
        f.write("3. Considerar implicações para estrutura atômica e molecular\n")
        f.write("4. Explorar conexões com propriedades do vácuo quântico\n")
        f.write("5. Avaliar potencial para unificação das constantes fundamentais\n")
        f.write("="*80 + "\n")
    
    print(f"📊 Relatório Constantes Espaço-Tempo salvo: {report_file}")
    return report_file

def infinite_hunter_spacetime():
    global shutdown_requested
    print(f"🚀 ZVT SPACETIME CONSTANTS RESONANCE HUNTER")
    print(f"🌡️ Constante de Boltzmann k_B = {K_BOLTZMANN:.3e} J/K")
    print(f"⚡ Carga Elementar e = {E_CHARGE:.3e} C")
    print(f"📡 Permissividade ε₀ = {EPSILON_0:.3e} F/m")
    print(f"🧲 Permeabilidade μ₀ = {MU_0:.3e} H/m")
    print(f"⚛️ Massa Elétron m_e = {M_ELECTRON:.3e} kg")
    print(f"🔬 Parâmetro Principal = {SPACETIME_CONSTANT:.3f}")
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
            print(f"    🎯 NOVO MELHOR GLOBAL ESPAÇO-TEMPO!")
            print(f"    Zero #{best_overall[0]:,} → γ mod (k_B×10²³) = {best_overall[2]:.15e}")
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
    
    print(f"\n📊 Gerando relatório final Constantes Espaço-Tempo...")
    generate_comprehensive_report(all_zeros, session_results, batch_num-1)
    return all_zeros, session_results, best_overall

def main():
    """Função principal"""
    global shutdown_requested
    print("🌟 Iniciando ZVT Spacetime Constants Resonance Hunter")
    
    try:
        zeros, results, best = infinite_hunter_spacetime()
        
        if zeros and len(zeros) > 0:
            print(f"\n🎯 Análise Constantes Espaço-Tempo Concluída!")
            print(f"📁 Resultados em: {RESULTS_DIR}/")
            print(f"💾 Cache: {CACHE_FILE}")
            print(f"📊 Estatísticas: {STATS_FILE}")
            
            if best:
                print(f"🏆 Melhor ressonância: Zero #{best[0]:,} com qualidade relativa {best[4]:.3e}")
                print(f"🌡️ Fator termodinâmico: γ = {best[1]/SPACETIME_CONSTANT:.3e} × k_B")
                print(f"🔬 Temperatura: T = {best[1]/(KB_SCALED*1e-23):.3e} K")
                print(f"⚡ Energia: E = {best[1]*1e-23:.3e} J")
            else:
                print(f"📊 Nenhuma ressonância excepcional encontrada")
                print(f"🤔 Zeros podem ser independentes das constantes espaço-tempo testadas")
                
    except KeyboardInterrupt:
        print(f"\n⏹️ Análise interrompida. Progresso salvo.")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n🔬 Sessão Constantes Espaço-Tempo concluída!")

if __name__ == "__main__":
    shutdown_requested = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
