#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_NUCLEAR_COSMIC_HUNTER.py - Zeta/Nuclear Forces & Cosmology resonance hunter
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

# Constantes das Forças Nucleares e Cosmologia
# Forças Nucleares
ALPHA_STRONG = 0.118                    # Constante de acoplamento forte (adimensional)
FERMI_CONSTANT = 1.1663787e-5          # GeV^-2 (força nuclear fraca)
QCD_SCALE = 0.217                      # GeV (escala QCD - Lambda_QCD)

# Constantes Cosmológicas  
HUBBLE_CONSTANT = 2.268e-18            # s^-1 (H₀ em unidades SI)
CMB_TEMPERATURE = 2.72548               # K (temperatura radiação cósmica)
COSMIC_LAMBDA = 1.1e-52                # m^-2 (constante cosmológica)
CRITICAL_DENSITY = 9.47e-27             # kg/m^3 (densidade crítica)

# Constantes do Modelo Padrão
HIGGS_MASS = 125.1                      # GeV (massa do bóson de Higgs)
WEINBERG_ANGLE = 0.23122                # ângulo de mistura eletrofraca (sin²θ_W)
W_BOSON_MASS = 80.379                   # GeV (massa do bóson W)
Z_BOSON_MASS = 91.1876                  # GeV (massa do bóson Z)
TOP_QUARK_MASS = 173.1                  # GeV (massa do quark top)

# Densidades cosmológicas (frações da densidade crítica)
OMEGA_MATTER = 0.315                    # fração de matéria
OMEGA_LAMBDA = 0.685                    # fração de energia escura
OMEGA_BARYON = 0.0493                   # fração de matéria bariônica
OMEGA_RADIATION = 9.2e-5                # fração de radiação

# Constantes já descobertas (para referência)
C_LIGHT = 299792458                     # m/s
H_PLANCK = 6.62607015e-34              # J⋅s
G_GRAVITY = 6.67430e-11                # m³⋅kg⁻¹⋅s⁻²
ALPHA = 7.2973525693e-3                # Estrutura fina
K_BOLTZMANN = 1.380649e-23             # J/K
EPSILON_0 = 8.8541878128e-12           # F/m

# Escalas de energia características
GEV_TO_JOULE = 1.602176634e-10         # Conversão GeV → J
PLANCK_MASS_GEV = 1.22e19              # GeV (massa de Planck)
ELECTROWEAK_SCALE = 246                 # GeV (vev do Higgs)

# Normalizações para escala dos zeros (~1-100)

# Grupo 1: Forças Nucleares
STRONG_SCALED = ALPHA_STRONG * 100      # ≈ 11.8 (força forte)
FERMI_SCALED = FERMI_CONSTANT * 1e5     # ≈ 1.17 (força fraca)
QCD_SCALED = QCD_SCALE * 10             # ≈ 2.17 (escala QCD)
WEINBERG_SCALED = WEINBERG_ANGLE * 100  # ≈ 23.1 (ângulo de Weinberg)

# Grupo 2: Massas do Modelo Padrão  
HIGGS_SCALED = HIGGS_MASS / 10          # ≈ 12.51 (Higgs)
W_MASS_SCALED = W_BOSON_MASS / 10       # ≈ 8.04 (W boson)
Z_MASS_SCALED = Z_BOSON_MASS / 10       # ≈ 9.12 (Z boson)
TOP_SCALED = TOP_QUARK_MASS / 10        # ≈ 17.31 (top quark)
ELECTROWEAK_SCALED = ELECTROWEAK_SCALE / 10  # ≈ 24.6 (escala eletrofraca)

# Grupo 3: Constantes Cosmológicas
HUBBLE_SCALED = HUBBLE_CONSTANT * 1e18  # ≈ 2.27 (Hubble)
CMB_SCALED = CMB_TEMPERATURE            # ≈ 2.73 (temperatura CMB)
LAMBDA_SCALED = COSMIC_LAMBDA * 1e52    # ≈ 1.1 (constante cosmológica)
CRITICAL_DENS_SCALED = CRITICAL_DENSITY * 1e27  # ≈ 9.47 (densidade crítica)

# Grupo 4: Densidades Cosmológicas
MATTER_SCALED = OMEGA_MATTER * 10       # ≈ 3.15 (fração matéria)
DARK_ENERGY_SCALED = OMEGA_LAMBDA * 10  # ≈ 6.85 (energia escura)
BARYON_SCALED = OMEGA_BARYON * 100      # ≈ 4.93 (matéria bariônica)
RADIATION_SCALED = OMEGA_RADIATION * 1e5 # ≈ 9.2 (radiação)

# Grupo 5: Relações Fundamentais
PLANCK_MASS_SCALED = PLANCK_MASS_GEV / 1e19  # ≈ 1.22 (massa Planck)
STRONG_WEAK_RATIO = ALPHA_STRONG / ALPHA  # ≈ 16.2 (razão forte/EM)
HIGGS_PLANCK_RATIO = HIGGS_MASS / PLANCK_MASS_GEV * 1e19  # ≈ 1.02e-17 → normalizar
COSMIC_TIME = 1 / HUBBLE_CONSTANT / (365.25 * 24 * 3600) / 1e9  # ≈ 14 Gyr → ≈ 14

# Constante principal para análise
NUCLEAR_COSMIC_CONSTANT = STRONG_SCALED  # Começando com força forte

TOLERANCE_LEVELS = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]  # Tolerâncias relativas
DEFAULT_TOLERANCE = 1e-8
INCREMENT = 1000  # Batch size for processing

# Constantes de controle para validação estatística
CONTROL_CONSTANTS = {
    # Grupo Forças Nucleares
    'strong_force': STRONG_SCALED,           # ≈ 11.8
    'weak_force': FERMI_SCALED,              # ≈ 1.17
    'qcd_scale': QCD_SCALED,                 # ≈ 2.17
    'weinberg_angle': WEINBERG_SCALED,       # ≈ 23.1
    'strong_em_ratio': STRONG_WEAK_RATIO,    # ≈ 16.2
    
    # Grupo Modelo Padrão
    'higgs_mass': HIGGS_SCALED,              # ≈ 12.51
    'w_boson': W_MASS_SCALED,                # ≈ 8.04
    'z_boson': Z_MASS_SCALED,                # ≈ 9.12
    'top_quark': TOP_SCALED,                 # ≈ 17.31
    'electroweak': ELECTROWEAK_SCALED,       # ≈ 24.6
    
    # Grupo Cosmologia
    'hubble_const': HUBBLE_SCALED,           # ≈ 2.27
    'cmb_temp': CMB_SCALED,                  # ≈ 2.73
    'lambda_cosmo': LAMBDA_SCALED,           # ≈ 1.1
    'critical_dens': CRITICAL_DENS_SCALED,   # ≈ 9.47
    'cosmic_age': COSMIC_TIME,               # ≈ 14
    
    # Grupo Densidades
    'dark_matter': MATTER_SCALED,            # ≈ 3.15
    'dark_energy': DARK_ENERGY_SCALED,       # ≈ 6.85
    'baryons': BARYON_SCALED,                # ≈ 4.93
    'radiation': RADIATION_SCALED,           # ≈ 9.2
    'planck_mass': PLANCK_MASS_SCALED,       # ≈ 1.22
    
    # Controles aleatórios
    'random_1': 11.5,
    'random_2': 2.5,
    'random_3': 24.8,
    'random_4': 14.2,
}

MAX_WORKERS = os.cpu_count()
CACHE_FILE = "zeta_zeros_cache.pkl"
STATS_FILE = "zvt_nuclear_cosmic_stats.txt"
RESULTS_DIR = "zvt_nuclear_cosmic_results"
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
        constants_dict = {'strong_force': NUCLEAR_COSMIC_CONSTANT}
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
        nuclear_results = find_multi_tolerance_resonances(zeros, {'strong_force': NUCLEAR_COSMIC_CONSTANT})['strong_force']
        tolerance_summary = {}
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in nuclear_results:
                resonances = nuclear_results[tolerance]
                count = len(resonances)
                rate = count / len(zeros) * 100 if len(zeros) > 0 else 0
                tolerance_summary[tolerance] = {
                    'count': count,
                    'rate': rate,
                    'best_quality': min(r[4] for r in resonances) if count > 0 else None,  # Erro relativo
                    'resonances': resonances,
                    'stats': None
                }
        print(f"\n📊 RESSONÂNCIAS BÁSICAS (FORÇAS NUCLEARES & COSMOLOGIA):")
        print("| Tolerância | Contagem | Taxa (%) |")
        print("|------------|----------|----------|")
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in tolerance_summary:
                data = tolerance_summary[tolerance]
                print(f"| {tolerance:8.0e} | {data['count']:8d} | {data['rate']:8.3f} |")
        return tolerance_summary, {}, None
    
    nuclear_results = find_multi_tolerance_resonances(zeros, {'strong_force': NUCLEAR_COSMIC_CONSTANT})['strong_force']
    print(f"\n📊 RESSONÂNCIAS FORÇAS NUCLEARES & COSMOLOGIA POR TOLERÂNCIA (α_s×100 = {NUCLEAR_COSMIC_CONSTANT:.3f}):")
    print("| Tolerância | Contagem | Taxa (%) | Melhor Qualidade | Significância |")
    print("|------------|----------|----------|------------------|---------------|")
    tolerance_summary = {}
    for tolerance in TOLERANCE_LEVELS:
        if tolerance in nuclear_results:
            resonances = nuclear_results[tolerance]
            count = len(resonances)
            rate = count / len(zeros) * 100
            if count > 0:
                best_quality = min(r[4] for r in resonances)  # Erro relativo
                stats_result = enhanced_statistical_analysis(zeros, resonances, NUCLEAR_COSMIC_CONSTANT, tolerance)
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
    
    print(f"\n🎛️ COMPARAÇÃO FORÇAS NUCLEARES & COSMOLOGIA (Tolerância: {DEFAULT_TOLERANCE}):")
    comparative_results = comparative_constant_analysis(zeros, DEFAULT_TOLERANCE)
    print("| Constante         | Valor          | Contagem | Taxa (%) | Significância |")
    print("|-------------------|----------------|----------|----------|---------------|")
    for const_name, results in comparative_results.items():
        count = results['resonance_count']
        rate = results['resonance_rate']
        sig_factor = results['stats']['basic_stats']['significance_factor'] if results['stats'] else 0
        value = results['constant_value']
        value_str = f"{value:.3f}"
        print(f"| {const_name:17s} | {value_str:14s} | {count:8d} | {rate:8.3f} | {sig_factor:8.2f}x |")
    
    best_overall = None
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['resonances']:
        best_overall = min(tolerance_summary[DEFAULT_TOLERANCE]['resonances'], key=lambda x: x[4])  # Por erro relativo
        print(f"\n💎 MELHOR RESSONÂNCIA NUCLEAR/CÓSMICA (Tolerância: {DEFAULT_TOLERANCE}):")
        print(f"   Zero #{best_overall[0]:,} → γ = {best_overall[1]:.15f}")
        print(f"   γ mod (α_s×100) = {best_overall[2]:.15e}")
        print(f"   Erro relativo: {best_overall[4]:.15e} ({best_overall[4]*100:.12f}%)")
        print(f"   Fator nuclear: γ = {best_overall[1]/NUCLEAR_COSMIC_CONSTANT:.6f} × (α_s×100)")
        print(f"   Força forte equivalente: α_s = {best_overall[1]/NUCLEAR_COSMIC_CONSTANT/100:.6f}")
        print(f"   Energia QCD: E = {best_overall[1]/NUCLEAR_COSMIC_CONSTANT/100 * QCD_SCALE:.3f} GeV")
        print(f"   Escala de confinamento: Λ_QCD = {QCD_SCALE:.3f} GeV")
    
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['stats']:
        stats = tolerance_summary[DEFAULT_TOLERANCE]['stats']
        print(f"\n📈 TESTES ESTATÍSTICOS NUCLEAR/CÓSMICOS (Tolerância: {DEFAULT_TOLERANCE}):")
        if 'chi2_test' in stats:
            chi2 = stats['chi2_test']
            print(f"   Qui-quadrado: χ² = {chi2['statistic']:.3f}, p = {chi2['p_value']:.6f}, Significativo: {'Sim' if chi2['significant'] else 'Não'}")
        if 'binomial_test' in stats:
            binom = stats['binomial_test']
            print(f"   Binomial: p = {binom['p_value']:.6f}, Significativo: {'Sim' if binom['significant'] else 'Não'}")
    
    return tolerance_summary, comparative_results, best_overall

def generate_comprehensive_report(zeros, session_results, final_batch):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(RESULTS_DIR, f"Relatorio_Nuclear_Cosmic_{timestamp}.txt")
    final_tolerance_analysis, final_comparative, final_best = analyze_batch_enhanced(zeros, final_batch)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ZVT NUCLEAR & COSMIC FORCES HUNTER - RELATÓRIO COMPREENSIVO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {datetime.now().isoformat()}\n")
        f.write(f"Versão: Nuclear & Cosmic Hunter v1.0\n")
        f.write(f"Lote Final: #{final_batch}\n")
        f.write(f"Zeros Analisados: {len(zeros):,}\n")
        f.write(f"Arquivo de Origem: {ZEROS_FILE}\n\n")
        f.write("CONFIGURAÇÃO FORÇAS NUCLEARES & COSMOLOGIA:\n")
        f.write(f"Constante de Acoplamento Forte (α_s): {ALPHA_STRONG:.15f}\n")
        f.write(f"Constante de Fermi (G_F): {FERMI_CONSTANT:.15e} GeV⁻²\n")
        f.write(f"Escala QCD (Λ_QCD): {QCD_SCALE:.15f} GeV\n")
        f.write(f"Ângulo de Weinberg (sin²θ_W): {WEINBERG_ANGLE:.15f}\n")
        f.write(f"Massa do Higgs: {HIGGS_MASS:.15f} GeV\n")
        f.write(f"Massa do Bóson W: {W_BOSON_MASS:.15f} GeV\n")
        f.write(f"Massa do Bóson Z: {Z_BOSON_MASS:.15f} GeV\n")
        f.write(f"Constante de Hubble (H₀): {HUBBLE_CONSTANT:.15e} s⁻¹\n")
        f.write(f"Temperatura CMB: {CMB_TEMPERATURE:.15f} K\n")
        f.write(f"Constante Cosmológica (Λ): {COSMIC_LAMBDA:.15e} m⁻²\n")
        f.write(f"Densidade Crítica: {CRITICAL_DENSITY:.15e} kg/m³\n")
        f.write(f"Tolerâncias: {TOLERANCE_LEVELS}\n")
        f.write(f"Tolerância Padrão: {DEFAULT_TOLERANCE}\n")
        f.write(f"Precisão: {mp.dps} casas decimais\n\n")
        f.write("ANÁLISE MULTI-TOLERÂNCIA (FORÇAS NUCLEARES & COSMOLOGIA):\n")
        f.write("| Tolerância | Ressonâncias | Taxa (%) | Melhor Qualidade | Significância |\n")
        f.write("|------------|--------------|----------|------------------|---------------|\n")
        for tolerance in TOLERANCE_LEVELS:
            if tolerance in final_tolerance_analysis:
                data = final_tolerance_analysis[tolerance]
                best_quality = "N/A" if data['best_quality'] is None else f"{data['best_quality']:.3e}"
                significance = "N/A" if data['significance'] == 0 else f"{data['significance']:8.2f}"
                f.write(f"| {tolerance:8.0e} | {data['count']:10d} | {data['rate']:8.3f} | {best_quality:>11s} | {significance:>8s}x |\n")
        
        f.write(f"\nCOMPARAÇÃO FORÇAS NUCLEARES & COSMOLOGIA (Tolerância: {DEFAULT_TOLERANCE}):\n")
        f.write("| Constante           | Valor              | Ressonâncias | Taxa (%) | Significância |\n")
        f.write("|---------------------|--------------------|--------------|----------|---------------|\n")
        for const_name, results in final_comparative.items():
            value_str = f"{results['constant_value']:.6f}"
            sig_factor = results['stats']['basic_stats']['significance_factor'] if results['stats'] else 0
            f.write(f"| {const_name:19s} | {value_str:18s} | {results['resonance_count']:10d} | {results['resonance_rate']:8.3f} | {sig_factor:8.2f}x |\n")
        
        if final_best:
            f.write(f"\nMELHOR RESSONÂNCIA NUCLEAR/CÓSMICA:\n")
            f.write(f"Índice do Zero: #{final_best[0]:,}\n")
            f.write(f"Valor Gamma: {final_best[1]:.20f}\n")
            f.write(f"Qualidade Absoluta: {final_best[2]:.20e}\n")
            f.write(f"Qualidade Relativa: {final_best[4]:.20e}\n")
            f.write(f"Erro Relativo: {final_best[4]*100:.12f}%\n")
            f.write(f"Fator Nuclear: {final_best[1]/NUCLEAR_COSMIC_CONSTANT:.6e}\n")
            f.write(f"α_s Equivalente: {final_best[1]/NUCLEAR_COSMIC_CONSTANT/100:.6f}\n")
            f.write(f"Energia QCD: {final_best[1]/NUCLEAR_COSMIC_CONSTANT/100 * QCD_SCALE:.6f} GeV\n")
            f.write(f"Relação Nuclear: γ = {final_best[1]/NUCLEAR_COSMIC_CONSTANT:.3e} × α_s\n\n")
        else:
            f.write(f"\nNENHUMA RESSONÂNCIA NUCLEAR/CÓSMICA ENCONTRADA nas tolerâncias testadas.\n\n")
        
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
        
        f.write("\nIMPLICAÇÕES PARA AS 4 FORÇAS FUNDAMENTAIS:\n")
        f.write("1. Investigar força nuclear forte e confinamento de quarks\n")
        f.write("2. Analisar força nuclear fraca e decaimentos radioativos\n")
        f.write("3. Explorar conexões com modelo padrão da física de partículas\n")
        f.write("4. Considerar implicações cosmológicas e estrutura em grande escala\n")
        f.write("5. Avaliar unificação completa das forças fundamentais\n")
        f.write("="*80 + "\n")
    
    print(f"📊 Relatório Nuclear & Cósmico salvo: {report_file}")
    return report_file

def infinite_hunter_nuclear_cosmic():
    global shutdown_requested
    print(f"🚀 ZVT NUCLEAR & COSMIC FORCES RESONANCE HUNTER")
    print(f"⚛️ Força Forte α_s = {ALPHA_STRONG:.6f}")
    print(f"☢️ Força Fraca G_F = {FERMI_CONSTANT:.3e} GeV⁻²")
    print(f"🌌 Hubble H₀ = {HUBBLE_CONSTANT:.3e} s⁻¹")
    print(f"🌡️ CMB T = {CMB_TEMPERATURE:.5f} K")
    print(f"🔬 Higgs = {HIGGS_MASS:.1f} GeV")
    print(f"⚡ W Boson = {W_BOSON_MASS:.3f} GeV")
    print(f"💫 Z Boson = {Z_BOSON_MASS:.4f} GeV")
    print(f"🎯 Parâmetro Principal = {NUCLEAR_COSMIC_CONSTANT:.3f}")
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
            print(f"    🎯 NOVO MELHOR GLOBAL NUCLEAR/CÓSMICO!")
            print(f"    Zero #{best_overall[0]:,} → γ mod (α_s×100) = {best_overall[2]:.15e}")
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
    
    print(f"\n📊 Gerando relatório final Nuclear & Cósmico...")
    generate_comprehensive_report(all_zeros, session_results, batch_num-1)
    return all_zeros, session_results, best_overall

def main():
    """Função principal"""
    global shutdown_requested
    print("🌟 Iniciando ZVT Nuclear & Cosmic Forces Resonance Hunter")
    
    try:
        zeros, results, best = infinite_hunter_nuclear_cosmic()
        
        if zeros and len(zeros) > 0:
            print(f"\n🎯 Análise Nuclear & Cósmica Concluída!")
            print(f"📁 Resultados em: {RESULTS_DIR}/")
            print(f"💾 Cache: {CACHE_FILE}")
            print(f"📊 Estatísticas: {STATS_FILE}")
            
            if best:
                print(f"🏆 Melhor ressonância: Zero #{best[0]:,} com qualidade relativa {best[4]:.3e}")
                print(f"⚛️ Fator nuclear: γ = {best[1]/NUCLEAR_COSMIC_CONSTANT:.3e} × α_s")
                print(f"🔬 α_s equivalente: {best[1]/NUCLEAR_COSMIC_CONSTANT/100:.6f}")
                print(f"💥 Energia QCD: {best[1]/NUCLEAR_COSMIC_CONSTANT/100 * QCD_SCALE:.3f} GeV")
            else:
                print(f"📊 Nenhuma ressonância excepcional encontrada")
                print(f"🤔 Zeros podem ser independentes das forças nucleares e cosmologia")
                
    except KeyboardInterrupt:
        print(f"\n⏹️ Análise interrompida. Progresso salvo.")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n🔬 Sessão Nuclear & Cósmica concluída!")

if __name__ == "__main__":
    shutdown_requested = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
