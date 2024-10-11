# -*- coding = utf-8 -*-
# @time:2023/10/8 15:56
# Author:Yuxiao
# @File:EpidemicSimulation.py
import time
from collections import defaultdict
from scipy.stats import lognorm
# from prob_model import prob_infectious_model
import numpy.lib.recfunctions as rfn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
import warnings
import pickle
import time
import sys
import os
import pdb
import warnings
from collections import deque
# 忽略所有警告消息
warnings.filterwarnings("ignore")
import torch

class EpidemicSimulation:

    # --- Different Level PHSM's Params --- #
    DICT_PHSM_PARAMS = [[1.0, 1.0, False, False, False, 0.00],
                        [1.0, 1.0, False, False, True,  0.10],
                        [0.5, 1.0, True,  False, True,  0.10],
                        [0.1, 1.0, True,  True,  True,  0.10],
                        [0.1, 1.0, True,  True,  True,  0.64]]
    # --- Compartments --- #
    SUSC:  int = 0 # 易感者
    LATT:  int = 1# 潜伏者
    INCB:  int = 2#被感染有感染力会发病的
    ASYM:  int = 3#无症状感染者
    SYMP:  int = 4#有症状感染者
    MILD:  int = 5# 轻症
    SEVE:  int = 6# 重症
    HOSP:  int = 7# 住院
    ICU:   int = 8
    HOSPR: int = 9#icu回了普通
    RECV:  int = 10#    治愈
    DEATH: int = 11#    死亡
    PROT:  int = 12#养老院保护
    OUT:   int = 13 #离开深圳的人

    COMP_NAMES: [str] = ["SUSC", "LATT", "INCB", "ASYM", "SYMP", "MILD", "SEVE", "HOSP", "ICU", "HOSPR", "RECV", "DEATH", "PROT", "OUT"]

    # --- Immune Status --- #
    V0: int = 0
    V1: int = 1
    V2: int = 2
    V3: int = 3

    # --- Occupation Type --- #
    HOME:   int = -1
    KINDER: int = 1
    PRIMARY:int = 2
    MIDDLE: int = 3
    HIGH:   int = 4
    WORK:   int = 5
    OTHER:  int = 6
    IMPORT: int = 7

    # --- Infectious Source Types --- #
    ST_ASYM = 0
    ST_PRES = 1
    ST_SYMP = 2

    # --- Personel Quarantine Level --- #
    Q_FREE  = 0
    Q_COM   = 1
    Q_ISO_HOME = 2
    Q_ISO  = 3
    Q_HOSP  = 4
    # Q_COM_FREE = 0
    # Q_COM_ISO = 1
    # Q_WORK_FREE = 0
    # Q_WORK_ISO = 1
    # Q_PCR   = 1  # 核酸检出隔离
    # Q_TRACK = 2  # 密接隔离
    # Q_ONSET = 3  # 轻症/就医门诊检出隔离
    # Q_ISO_HOME = 4  # 自身发病居家
    # Q_ISO_H = 3  # 家人发病居家
    # Q_HOSP  = 4  # 住院（视作隔离）
    # Q_DISCH = 4  # 出院后的观察隔离期
    # Q_CLASS = 4  # 班级停课
    # Q_CABIN = 4  # 方舱隔离

    days_school_off = 27             # 中小学放寒假
    days_school_on  = 65             # 中小学开学


    def __init__(
        self,
        city_name: str,
        ptrans: float,

        p_mask: float = 0.80,                         # 口罩佩戴比例
        e_mask: float = 0.50,                         # 口罩的有效性 (仅作用于公共场所)
        p_drug: float = 0.00,                         # 药物供给比例
        e_drug: float = 0.24,                         # 药物效用 (减少入院比例)
        e_drug_death: float = 0.55,                   # 药物效用（减少死亡比例)
        priority_drug: str = "Mortality",             # 药物供给优先组 (Mortality/Eldly/Random 死亡风险/年龄/随机)

        init_level: int = -1,                         # 初始的npi级别
        low_level: int = 1,                           # 高限制60d/ICU降至阈值以下/总人群超50%已感染后的npi级别
        high_level: int = 1,                          # ICU占用超阈值后的npi级别
        th_icu_on: float = np.inf,                    # 触发各类的ICU占用率阈值
        th_icu_off: float = np.inf,                   # 解除各类的ICU占用率阈值
        is_pop_off: bool = False,                     # 是否通过感染规模超过50%接触限制，False：通过th_icu_off，True：通过感染规模
        th1: float = np.inf,
        th2: float = np.inf,
        th3: float = np.inf,
        th4: float = np.inf,

        rate_iso_p: float = 0.00,                     # 个人居家比例
        rate_iso_h: float = 0.00,                     # 全家居家比例
        days_iso_p: float = 10.00,                    # 个人居家时长
        days_iso_h: float = 10.00,                    # 全家居家时长
        theta_iso:  float = 0.70,                     # 居家过程的家庭感染强度降低至原家庭感染强度的比例
        fixed_r_symp: float = 0.80,                   # 当 fixed_r_symp>0 时, 以fixed_r_symp的值作为全年龄段的显性感染者占比 (反之使用上海分年龄数据)
        days_iso_discharge: float = 0.00,             # 出院之后需要隔离时长

        r_ic_other: float = 1.0,                      # 实施娱乐场所限流后保留的比例, 通过降低ic实现
        r_ic_work: float = 1.0,                       # 实施工作场所限流后保留的比例, 通过降低ic实现
        r_ic_school: float = 1.0,                     # 实施中小学限流后保留的比例, 通过降低ic实现, r_ic_school=0时学校停课

        is_close_class: bool = False,                 # 是否实行班级停课
        th_close_class: int = 5,                      # 触发班级停课的发病人数阈值
        rate_onset_report: float = 0.25,              # 轻症发病学生上报比例 (重症必上报)
        days_close_class: float = 10.00,              # 班级停课时长
        is_npi_gather: bool = False,                  # 是否限制集会
        npi_max_gather: int = 10,                     # 限制集会的最大人数

        th_icu_school_off: float = np.inf,            # 触发停学的ICU占用率阈值
        th_icu_school_on: float = np.inf,             # 解除停学的ICU占用率阈值
        th_icu_work_off: float = np.inf,              # 触发停工的ICU占用率阈值
        th_icu_work_on: float = np.inf,               # 解除停工的ICU占用率阈值
        rate_iso_p_work_off: float = 0.6,             # 触发停工后的asym个人居家比例
        rate_iso_h_work_off: float = 1.0,             # 触发停工后的asym全家居家比例
        is_elderly_protect: bool = False,             # 是否反向保护老年人
        n_imported_daily: int = 1,                    # 每日固定输入种子,以模拟大流行中的跨城输入,避免群体免疫前感染力个体清零

        is_spatial: bool = True,                      # 是否采用空间显式模型
        is_samp: bool = False,                        # 使用采样网络 (仅用于测试、调试和预实验)
        samp_rate: float = 0.1,                       # 采样网络的采样比例

        is_fast_r0: bool = False,                     # 使用感染日期计算R0
        is_reinfect: bool = True,                     # 个体是否能够被再次感染   True-SIRS  False:SIR
        is_ve_anti_infection: bool = True,            # 疫苗是否具备防感染能力
        uptake_scenario: str = "Uptake 90",           # 四种疫苗接种场景: Uptake 00, Uptake Current, Uptake 90, Uptake Enhence
        list_imported_daily: list = [],               # 指定的输入种子列表: 输入日期-输入种子年龄-所在行政区-所在街道-所在社区 (用于现实场景流调数据疫情建模)
        n_imported_day0: int = 30,                    # 在未指定种子列表时, 设定第0天感染的种子数
        max_iterday: int = 365,                       # 模拟总天数
        duration_sigma: float = 0.1,                  # 仓室转移时间的标准差
        seed:int=1,
        path_param=r"/home/yxluo/research/EPC_RF/env/EPC_env/Parameter/",
        path_prop=r"/home/yxluo/research/EPC_RF/env/EPC_env/Property/",
        path_data=r"/home/yxluo/research/EPC_RF/env/EPC_env/Data/",
        ) -> None:

        if samp_rate>=1.0:
            is_samp = False  # 采样率达到100%即完整网络

        self.city_name = city_name

        # modified by LX
        self.init_level = init_level
        self.low_level = low_level
        self.high_level = high_level
        self.temp_level = self.init_level
        self.levels = []
        self.is_pop_off = is_pop_off
        self.th_icu_on = th_icu_on
        self.th_icu_off = th_icu_off

        # modified at 20221115
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.th4 = th4
        self.temp_phase = 0

        self.ptrans = ptrans
        self.p_mask = p_mask
        self.e_mask = e_mask
        self.p_drug = p_drug
        self.e_drug = e_drug
        self.e_drug_death = e_drug_death
        self.priority_drug = priority_drug

        self.rate_iso_p = rate_iso_p
        self.rate_iso_h = rate_iso_h
        self.days_iso_p = days_iso_p
        self.days_iso_h = days_iso_h
        self.theta_iso = theta_iso
        self.days_iso_discharge = days_iso_discharge

        self.r_ic_other = r_ic_other
        self.r_ic_work = r_ic_work
        self.r_ic_school = r_ic_school

        self.is_close_class = is_close_class
        self.th_close_class = th_close_class
        self.rate_onset_report = rate_onset_report
        self.days_close_class = days_close_class
        self.is_npi_gather = is_npi_gather
        self.npi_max_gather = npi_max_gather

        self.th_icu_work_on = th_icu_work_on
        self.th_icu_work_off = th_icu_work_off
        self.th_icu_school_on = th_icu_school_on
        self.th_icu_school_off = th_icu_school_off
        self.rate_iso_p_work_off = rate_iso_p_work_off
        self.rate_iso_h_work_off = rate_iso_h_work_off
        self.is_elderly_protect = is_elderly_protect
        self.n_imported_daily = n_imported_daily

        self.is_spatial = is_spatial
        self.is_samp = is_samp
        self.samp_rate = samp_rate

        self.is_fast_r0 = is_fast_r0
        self.is_reinfect = is_reinfect
        self.is_ve_anti_infection = is_ve_anti_infection
        self.uptake_scenario = uptake_scenario
        self.list_imported_daily = list_imported_daily
        self.n_imported_day0 = n_imported_day0
        self.max_iterday = max_iterday
        self.duration_sigma = duration_sigma

        self.path_param = path_param
        self.path_prop = path_prop
        self.path_data = path_data

        self.n_age_group = 13
        self.date_pid_imported = []  # 感染种子记录

        self._set_contact()
        self._set_trans_rate()
        self._set_compartment_duration()
        self._check_parameters_input()
        self._load_city_info()
        self._load_agents()
        self._init_npi_level()
        self.set_PHSM_params()
        self.no_more_change = False
        self.off_allowance = False
        self.icu_peak_arrived = False
        self.level4days = 0

        # np.random.seed(seed)


    def set_PHSM_params(self):
        if self.temp_level==-1:
            return

        r_ic_other, r_ic_work, is_npi_gather, is_school_off, is_iso_h, rate_iso_p_work_off = self.DICT_PHSM_PARAMS[self.temp_level]

        self.r_ic_other = r_ic_other
        self.r_ic_work = r_ic_work
        self.is_npi_gather = is_npi_gather
        self.rate_iso_p_work_off = rate_iso_p_work_off
        self.r_ic_school = 0 if is_school_off else 1
        self.th_icu_work = 0 if is_iso_h else np.inf

    def simulate_elderly_protection(self, sim_mat, elderly_group=[11,12]):
        if self.is_elderly_protect:
            pid_elderly = self.tab_person["pid"][np.isin(self.tab_person["age"], elderly_group)]
            self.n_elderly_prot = min(self.n_elderly_prot, len(pid_elderly))
            self.pid_elderly_protect = np.random.choice(pid_elderly, self.n_elderly_prot, replace=False)
            sim_mat["comp_this"][self.pid_elderly_protect] = self.PROT
            sim_mat["cd_trans" ][self.pid_elderly_protect] = np.inf
        else:
            self.pid_elderly_protect = np.empty(shape=0, dtype=int)
        self.pid_unprotect = np.setdiff1d(sim_mat["pid"], self.pid_elderly_protect)
        return sim_mat

    def _set_contact(self):
        """
        接触参数初始化:
        1. Workplace和Education网络上 社团内/社团间 的接触数量
        2. 不同source状态(发病前/发病后/无症状) 的相对感染力
        3. 不同活动场所(网络)的接触强度ic  公共场所(community网络)的ic需叠加口罩效用
        """
        self.n_contact_wgroup = {self.PRIMARY:20, self.MIDDLE:20, self.HIGH:20, self.KINDER:10, self.WORK:7}
        self.n_contact_wplace = {self.PRIMARY:5, self.MIDDLE:5, self.HIGH:5, self.KINDER:2, self.WORK:3}
        self.infectivity = {self.ST_ASYM:1, self.ST_PRES:1, self.ST_SYMP:1}

        self.ic_setting = {self.HOME:0.37, self.KINDER:0.25, self.PRIMARY:0.25, self.MIDDLE:0.25, self.HIGH:0.25, self.WORK:0.26, self.OTHER:0.1}
        # self.ic_setting[self.OTHER  ] = self.ic_setting[self.OTHER  ]*(1-self.p_mask*self.e_mask)
        # self.ic_setting[self.WORK   ] = self.ic_setting[self.WORK   ]*self.r_ic_work
        # self.ic_setting[self.KINDER ] = self.ic_setting[self.KINDER ]*self.r_ic_school
        # self.ic_setting[self.PRIMARY] = self.ic_setting[self.PRIMARY]*self.r_ic_school
        # self.ic_setting[self.MIDDLE ] = self.ic_setting[self.MIDDLE ]*self.r_ic_school
        # self.ic_setting[self.HIGH   ] = self.ic_setting[self.HIGH   ]*self.r_ic_school

    def _set_trans_rate(self):
        """
        从pkl文件中读取并初始化仓室转移概率, 发病率选择上海/深圳参数
        计算不同疫苗免疫类型 和 衰减天数 的仓室转移概率
        """
        trans_rate_v0 = pickle.load(open(os.path.join(self.path_param, "Epidemiological/", "TransitionRate-Adj.pkl"), "rb"))                     # <-----------------
        vaccine_efficacy = pd.read_excel(os.path.join(self.path_param, "Ref Parameter/VE-conditional-统一参数.xls"),index_col="dose")             # <-----------------
        self.trans_rate = self._set_vaccine_efficacy_waning(trans_rate_v0, vaccine_efficacy)

    def _set_vaccine_efficacy_waning(self, trans_rate_v0, vaccine_efficacy, method="linear"):
        """
        计算疫苗衰减下的仓室转移概率
        """
        trans_rate = defaultdict(dict)
        self.max_VE_span = max(vaccine_efficacy["span"])
        vaccine_efficacy["min_ve_inf"  ] = vaccine_efficacy["ve_inf"  ]*vaccine_efficacy["r_min_ve_inf"  ]
        vaccine_efficacy["min_ve_symp" ] = vaccine_efficacy["ve_symp" ]*vaccine_efficacy["r_min_ve_symp" ]
        vaccine_efficacy["min_ve_hosp" ] = vaccine_efficacy["ve_hosp" ]*vaccine_efficacy["r_min_ve_hosp" ]
        vaccine_efficacy["min_ve_death"] = vaccine_efficacy["ve_death"]*vaccine_efficacy["r_min_ve_death"]

        for dose, row in vaccine_efficacy.iterrows():
            vaccine_efficacy_symp  = np.r_[np.linspace(row.ve_symp, row.min_ve_symp, int(row.span)), np.full(int(self.max_VE_span-int(row.span)), row.min_ve_symp )]
            vaccine_efficacy_hosp  = np.r_[np.linspace(row.ve_hosp, row.min_ve_hosp, int(row.span)), np.full(int(self.max_VE_span-int(row.span)), row.min_ve_hosp )]
            vaccine_efficacy_death = np.r_[np.linspace(row.ve_death,row.min_ve_death,int(row.span)), np.full(int(self.max_VE_span-int(row.span)), row.min_ve_death)]
            vaccine_efficacy_inf = np.r_[np.linspace(row.ve_inf, row.min_ve_inf, int(row.span_inf)), np.full(int(self.max_VE_span-int(row.span)), row.min_ve_inf)]
            vaccine_efficacy_inf = np.pad(vaccine_efficacy_inf, (0, int(self.max_VE_span-row.span_inf)), mode="edge")

            for age in range(self.n_age_group):
                # set transition rate value with different vaccine status and age gourp
                trans_rate["susc2latt" ][(dose, age)] = trans_rate_v0["susc2latt" ][age]*(1-vaccine_efficacy_inf  )
                trans_rate["incb2symp" ][(dose, age)] = trans_rate_v0["incb2symp" ][age]*(1-vaccine_efficacy_symp )
                trans_rate["symp2seve" ][(dose, age)] = trans_rate_v0["symp2seve" ][age]*(1-vaccine_efficacy_hosp )
                trans_rate["hosp2death"][(dose, age)] = trans_rate_v0["hosp2death"][age]*(1-vaccine_efficacy_death)
                trans_rate["icu2death" ][(dose, age)] = trans_rate_v0["icu2death" ][age]*(1-vaccine_efficacy_death)
                trans_rate["seve2icu"  ][(dose, age)] = np.repeat(trans_rate_v0["seve2icu"][age], self.max_VE_span)
        return trans_rate

    def _plot_vaccine_efficacy_waning(self, method="linear"):
        """
        绘制不同针剂疫苗在不同作用节点的衰减曲线
        """
        plt.rcParams['font.sans-serif'] = 'SimHei'
        vaccine_efficacy = pd.read_excel(os.path.join(self.path_param, "Ref Parameter/VE setting.xls"), index_col="dose")

        self.max_VE_span = max(vaccine_efficacy["span"])
        vaccine_efficacy["min_ve_inf"  ] = vaccine_efficacy["ve_inf_" ]*vaccine_efficacy["r_min_ve_inf"  ]
        vaccine_efficacy["min_ve_symp" ] = vaccine_efficacy["ve_symp" ]*vaccine_efficacy["r_min_ve_symp" ]
        vaccine_efficacy["min_ve_hosp" ] = vaccine_efficacy["ve_hosp" ]*vaccine_efficacy["r_min_ve_hosp" ]
        vaccine_efficacy["min_ve_death"] = vaccine_efficacy["ve_death"]*vaccine_efficacy["r_min_ve_death"]

        ve = defaultdict(dict)
        for dose, row in vaccine_efficacy.iterrows():
            ve["VE against infection"      ][dose] = np.pad(np.linspace(row.ve_inf_, row.min_ve_inf,  int(row.span)),(0,int(self.max_iterday-row.span)),"edge")
            ve["VE against symptomatic"    ][dose] = np.pad(np.linspace(row.ve_symp, row.min_ve_symp, int(row.span)),(0,int(self.max_iterday-row.span)),"edge")
            ve["VE against hospitalization"][dose] = np.pad(np.linspace(row.ve_hosp, row.min_ve_hosp, int(row.span)),(0,int(self.max_iterday-row.span)),"edge")
            ve["VE against mortality      "][dose] = np.pad(np.linspace(row.ve_death,row.min_ve_death,int(row.span)),(0,int(self.max_iterday-row.span)),"edge")

    def _set_compartment_duration(self):
        """
        读取并初始化仓室转移时间
        是否能够二次感染
        """
        self.compart_duration = pd.read_excel(os.path.join(self.path_param, "Ref Parameter/compartment duration-Adj.xls"), index_col="Endpoint").to_dict()["Duration"]      # <--------
        if not self.is_reinfect:
            self.compart_duration["recv2susc"] = self.max_iterday+1

    def _set_compartment_structure(self):
        """
        UDC: User-Defined Compartment    未实装
        """
        self.compart_struct = {}
        self.compart_struct[self.LATT] = [self.INCB, None,      "latt2incb", None       ]
        self.compart_struct[self.INCB] = [self.SYMP, self.ASYM, "incb2symp", "incb2asym"]
        self.compart_struct[self.ASYM] = [self.RECV, None,      "asym2recv", None       ]

    def _init_npi_level(self):
        """初始状态下不停工停学"""
        self.is_work_off = False
        self.is_school_off = False

    def _get_vaccination_coverage(self):
        """
        读取分年龄组的疫苗接种率
        """
        fn_uptake_age = os.path.join(self.path_data, "疫苗接种率数据/Uptake Rate %s.xls"%(self.city_name))
        tab_uptake_age = pd.read_excel(fn_uptake_age).drop(columns=["Dose"])
        return tab_uptake_age

    def _get_age_specified_motality_risk(self):
        """
        计算重症患者的死亡风险 (无疫苗状态), 以支持药物优先供给策略
        """
        trans_rate_v0 = pickle.load(open(os.path.join(self.path_param, "Epidemiological/", "TransitionRate.pkl"), "rb"))
        r_seve2icu    = np.array([trans_rate_v0["seve2icu"  ][age] for age in range(self.n_age_group)])
        r_icu2death   = np.array([trans_rate_v0["icu2death" ][age] for age in range(self.n_age_group)])
        r_hosp2death  = np.array([trans_rate_v0["hosp2death"][age] for age in range(self.n_age_group)])
        motality_risk = r_icu2death*r_seve2icu+r_hosp2death*(1-r_seve2icu)
        return motality_risk

    def _check_parameters_input(self):
        """
        输入参数校验:
        1. 各场所的 ptran*ic 均不超过100% (单次接触产生感染的可能性SAR)
        2. 检验疫苗场景输入 - 无疫苗/深圳当前接种率/3岁以上90接种率
        3. 输入种子列表 或 输入初始种子数   两者择其一
        4. 检验药物供给策略输入
        5. 仓室时间无不确定时 (用于模型测试) 抛出警告提醒
        6. 执行密接时，检验密接概率分布之和是否等于100%
        """
        # assert self.ptrans*max(self.ic_setting.values())*max(self.infectivity.values())<=1, "Secondary Attack Rate>1!"
        assert self.uptake_scenario in ["Uptake 00", "Uptake Current", "Uptake 90", "Uptake Enhence", "Uptake Enhence 60", "Uptake Enhence Homo", "Uptake Enhence 60 Homo"], "Wrong Input of Uptake Scenario!"
        assert bool(len(self.list_imported_daily))^bool(self.n_imported_day0), "Either Input the List or the Number of Imported Seeds!"
        assert self.priority_drug in ["Mortality", "Eldly", "Random"], "Wrong Input of Drug Priority!"
        if self.duration_sigma<0.01:
            warnings.warn("Duration Sigma<0.01, May Lead to Unexpected Simulation Result.")

    def _load_city_info(self):
        city_info = pd.read_excel(os.path.join(self.path_param, "Hospital Capacity/典型城市基本信息.xls"))
        city_info = city_info.loc[city_info["city_name"]==self.city_name]
        self.n_city_pop = city_info["人口总数"].values[0]
        self.n_elderly_prot = city_info["养老院人数"].values[0]
        self.n_cap_hosp = city_info["可用普通床位数量"].values[0]
        self.n_cap_icu  = city_info["可用ICU床位数量"].values[0]
        self.label_city = city_info["城市"].values[0]

    def _fast_load_structured_array(self, fn_person):
        """
        由于底层优化方式不同, pandas读取csv再转换为structured_array在速度上显著优于np.genfromtxt
        ref: https://stackoverflow.com/questions/55577256
        """
        col_names = ['pid','hid','age','hzone','wzone','wtype','wplace','wgroup','nco_sq','nco_jd','h_jd']
        # col_names = ['pid','hid','age','hzone','wzone','wtype','wplace','wgroup','nco','hzone','tag']
        # tab_person = pd.read_csv(fn_person)[col_names]
        # tab_person["day_l"] = np.random.uniform(35, 42, tab_person.shape[0])
        # tab_person["day_r"] = np.random.uniform(43, 50, tab_person.shape[0])
        # tab_person.loc[tab_person.tag==0, "day_l"] = -1
        # tab_person.loc[tab_person.tag==0, "day_r"] = -1
        # tab_person.to_csv(fn_person, index=False)
        # exit()
        tab_person_df=pd.read_csv(fn_person)
        tab_person_df["ranked_wplace"] = (tab_person_df['wplace'].rank(method='dense') - 1).astype(int)
        tab_person_df["ranked_wgroup"] = (tab_person_df['wgroup'].rank(method='dense') - 1).astype(int)
        tab_person_df["ranked_hid"]  = (tab_person_df['hid'].rank(method='dense') - 1).astype(int)
        tab_person_df["ranked_hzone"] = (tab_person_df['hzone'].rank(method='dense') - 1).astype(int)
        tab_person_df["ranked_jdzone"] = (tab_person_df['h_jd'].rank(method = 'dense') - 1).astype(int)
        # new_row_data = {
        #     'pid':len(tab_person_df),
        #     'hid':0,
        #     'age':0,
        #     'gender':0,
        #     'hzone':0,
        #     'wzone':0,
        #     'wtype':0,
        #     'wplace':-1,
        #     'wgroup':0,
        #     'ozone':0,
        #     'n_c_h':0,
        #     'n_c_w':0,
        #     'n_c_o_exp':0,
        #     'ogroup':0,
        #     'n_c_o':0
        # }
        # # 在DataFrame的末尾添加新行
        # new_row_df = pd.DataFrame([new_row_data])
        # # 将新行DataFrame与原有DataFrame合并
        # tab_person_df = pd.concat([tab_person_df, new_row_df], ignore_index = True)
        tab_person = tab_person_df[col_names].to_numpy()
        p_dtype = np.dtype([("pid", "int32"), ("hid", "int32"), ("age", "int8"),
                            ("h_sq", "int64"), ("w_sq", "int64"),
                            ("wtype", "int8"), ("wplace", "int32"),
                            ("wgroup", "int32"), ("nco_sq", "int32"), ("nco_jd", "int32"), ("h_jd", "int64")])
        tab_person_struct = rfn.unstructured_to_structured(tab_person, p_dtype)
        self.tab_person_rank=tab_person_df[['pid','hid','age','hzone','wzone','wtype','wplace','wgroup',"ranked_wplace","h_jd","ranked_wgroup","ranked_hid","ranked_hzone","ranked_jdzone"]]
        # tab_model= tab_person_df[['pid','hid','age','hzone','wzone','wtype','wplace','wgroup',"ranked_wplace","ranked_wgroup","ranked_hid","ranked_hzone"]]

        self.jd_sq_marix = np.vstack(tab_person_df.groupby('ranked_hzone')['ranked_jdzone'].unique().to_numpy()).squeeze(axis=1)


        return tab_person_struct,tab_person_df

    def _group_agents(self, fn_person):

        fn_person_wplace_group = fn_person[fn_person['wtype'] == 5].groupby('wplace')
        fn_person_wgroup_group = fn_person[fn_person['wtype'] == 5].groupby('wgroup')
        fn_person_school_group = fn_person[(fn_person['wtype'] < 5) & (fn_person['wtype'] > 0)].groupby('wplace')
        fn_person_class_group = fn_person[(fn_person['wtype'] < 5) & (fn_person['wtype'] > 0)].groupby('wgroup')
        fn_person_com_group = fn_person.groupby('hzone')
        fn_person_home_group = fn_person.groupby('hid')
        return fn_person_wplace_group, fn_person_wgroup_group, fn_person_school_group, fn_person_class_group, fn_person_com_group, fn_person_home_group
    def _load_agents(self):
        """
        读取完整/采样的人工人口列表  生成静态属性表 tab_person
        如使用采样网络, 计算缩放因子
        集会限制策略作用于人口列表: 限制nco不超过阈值
        如进行小范围动态停学(班级), 预计算所有学生ID
        """
        fn_person = "".join([self.city_name]+["/synpop"    ]+["_spatial" if self.is_spatial else "_nonspatial"]+["_samp_%0.2f" % (self.samp_rate) if self.is_samp else ""]+["_new.csv"       ])
        fn_hdict  = "".join([self.city_name]+["/colocation"]+["_spatial" if self.is_spatial else "_nonspatial"]+["_samp_%0.2f" % (self.samp_rate) if self.is_samp else ""]+["_hdict.pkl" ])
        fn_hzdict = "".join([self.city_name]+["/colocation"]+["_spatial" if self.is_spatial else "_nonspatial"]+["_samp_%0.2f" % (self.samp_rate) if self.is_samp else ""]+["_hzdict.pkl"])
        fn_hjdict = "".join([self.city_name]+["/colocation"]+["_spatial" if self.is_spatial else "_nonspatial"]+["_samp_%0.2f" % (self.samp_rate) if self.is_samp else ""]+["_hjdict.pkl"])
        fn_wpdict = "".join([self.city_name]+["/colocation"]+["_spatial" if self.is_spatial else "_nonspatial"]+["_samp_%0.2f" % (self.samp_rate) if self.is_samp else ""]+["_wpdict.pkl"])
        fn_wgdict = "".join([self.city_name]+["/colocation"]+["_spatial" if self.is_spatial else "_nonspatial"]+["_samp_%0.2f" % (self.samp_rate) if self.is_samp else ""]+["_wgdict.pkl"])

        # fn_person = "".join([self.city_name]+["/synpop"    ]+["_spatial" if self.is_spatial else "_nonspatial"]+["_samp_%0.2f" % (self.samp_rate) if self.is_samp else ""]+["_spring.csv"])

        fn_person = os.path.join(self.path_prop, "Synthetic Population/", fn_person)
        fn_hdict  = os.path.join(self.path_prop, "Colocation/",           fn_hdict )
        fn_hzdict = os.path.join(self.path_prop, "Colocation/",           fn_hzdict)
        fn_hjdict = os.path.join(self.path_prop, "Colocation/",           fn_hjdict)
        fn_wpdict = os.path.join(self.path_prop, "Colocation/",           fn_wpdict)
        fn_wgdict = os.path.join(self.path_prop, "Colocation/",           fn_wgdict)
        self.tab_person,self.tab_df = self._fast_load_structured_array(fn_person)
        self.n_agent = self.tab_person.shape[0]

        self.scale_factor = self.n_city_pop/self.n_agent

        with open(fn_hdict,  "rb") as fh_hdict,  \
             open(fn_hzdict, "rb") as fh_hzdict, \
             open(fn_hjdict, "rb") as fh_hjdict, \
             open(fn_wpdict, "rb") as fh_wpdict, \
             open(fn_wgdict, "rb") as fh_wgdict:
            self.hdict  = pickle.load(fh_hdict)
            self.hzdict = pickle.load(fh_hzdict)
            self.hjdict = pickle.load(fh_hjdict)
            self.wpdict = pickle.load(fh_wpdict)
            self.wgdict = pickle.load(fh_wgdict)

        self.tab_person["nco_sq"] = self.tab_person["nco_sq"]*2
        self.tab_person["nco_jd"] = self.tab_person["nco_jd"] * 2

        self.nco_sq_init = self.tab_person["nco_sq"].copy()
        self.nco_jd_init = self.tab_person["nco_jd"].copy()


        # if self.is_npi_gather:
        #     np.putmask(self.tab_person["nco_sq"], self.tab_person["nco_sq"]>self.npi_max_gather, self.npi_max_gather)                     # <--------------------

        # self.tab_person['wtype']=-1
        self.pid_students = np.where(np.isin(self.tab_person["wtype"], [self.KINDER, self.PRIMARY, self.MIDDLE, self.HIGH]))[0]
        self.pid_workers = np.where(np.isin(self.tab_person["wtype"], [self.WORK]))[0]
        self.pid_home=np.where(np.isin(self.tab_person["wtype"], [self.HOME]))[0]
    def _random_binomial_choice(self, pids, prob, replace=False):
        """
        从候选个体集中按给定概率进行二项选择
        """
        prob = min(prob, 1)
        n_pid = np.random.binomial(len(pids), prob)
        pids_choice = np.random.choice(pids, n_pid, replace=replace)
        return pids_choice

    def __get_metadata__(self):
        """
        获取实验参数设置
        """
        metadata = self.__dict__.copy()
        drop_keys = ["hdict", "hzdict", "wpdict", "wgdict", "tab_person", "trans_rate", "gdf_sq", "dict_qu_jd", "dict_jd_sq"]
        drop_keys = np.intersect1d(drop_keys, list(metadata.keys()))
        for k in drop_keys:
            del metadata[k]
        return metadata

    def init_sim_mat(self):
        """
        初始化动态表: 包括 [个体ID,当前仓室,下一仓室,仓室转移倒计时,免疫类型,免疫衰减天数,已被感染的天数,隔离类型,隔离倒计时,药物可及性]
        在深圳动态清零(三区防控)模型中 增加 密接控制倒计时(整数型)
        """
        person_ids = self.tab_person["pid"]
        this_compartment     = np.repeat(self.SUSC,   self.n_agent)
        next_compartment     = np.repeat(self.SUSC,   self.n_agent)
        transit_countdown    = np.repeat(np.inf,      self.n_agent)
        immunity_type        = np.repeat(self.V0,     self.n_agent)
        immunity_days        = np.repeat(0,           self.n_agent)
        infected_days        = np.repeat(np.inf,      self.n_agent)
        quarantine_type      = np.repeat(self.Q_FREE, self.n_agent)
        quarantine_countdown = np.repeat(np.inf,      self.n_agent)
        drug_access          = np.repeat(False,       self.n_agent)
        # if_control           = np.repeat(True,       self.n_agent)
        # padding =np.concatenate(([[self.n_agent]],np.zeros((1,9))),axis=-1)

        sim_mat = np.c_[person_ids, this_compartment, next_compartment, transit_countdown, immunity_type, immunity_days,
                        infected_days, quarantine_type, quarantine_countdown, drug_access]
        # sim_mat=np.concatenate((sim_mat,padding),axis = 0)

        s_dtype = np.dtype([("pid", "int32"), ("comp_this", "int8"), ("comp_next", "int8"), ("cd_trans", "float"),
                            ("type_imune", "int8"),("days_imune", "int16"), ("days_infect", "int16"), ("type_quran", "int8"),
                            ("cd_quran", "float"), ("ac_drug", "bool")])
        sim_mat = rfn.unstructured_to_structured(sim_mat, s_dtype)
        return sim_mat


    def get_stat_features(self):
        df_onehot = self.tab_df
        self.wp_num = df_onehot.groupby(['wplace'])['pid'].transform('count').to_numpy().reshape(-1,1)
        self.wg_num = df_onehot.groupby(['wgroup'])['pid'].transform('count').to_numpy().reshape(-1,1)
        self.home_num = df_onehot.groupby(['hid'])['pid'].transform('count').to_numpy().reshape(-1,1)
        self.com_num = df_onehot.groupby(['hzone'])['pid'].transform('count').to_numpy().reshape(-1,1)
        self.h_jd_num = df_onehot.groupby(['h_jd'])['pid'].transform('count').to_numpy().reshape(-1,1)
        # self.n_contact_wgroup = {self.PRIMARY:20, self.MIDDLE:20, self.HIGH:20, self.KINDER:10, self.WORK:7}
        # self.n_contact_wplace = {self.PRIMARY:5, self.MIDDLE:5, self.HIGH:5, self.KINDER:2, self.WORK:3}

        self.stat_prob = np.zeros(len(self.tab_df))
        self.stat_prob[self.tab_df['wtype'] == self.PRIMARY]= self.n_contact_wgroup[self.PRIMARY] + self.n_contact_wplace[self.PRIMARY]
        self.stat_prob[self.tab_df['wtype'] == self.MIDDLE]= self.n_contact_wgroup[self.MIDDLE] + self.n_contact_wplace[self.MIDDLE]
        self.stat_prob[self.tab_df['wtype'] == self.HIGH]= self.n_contact_wgroup[self.HIGH] + self.n_contact_wplace[self.HIGH]
        self.stat_prob[self.tab_df['wtype'] == self.KINDER]= self.n_contact_wgroup[self.KINDER] + self.n_contact_wplace[self.KINDER]
        self.stat_prob[self.tab_df['wtype'] == self.WORK]= self.n_contact_wgroup[self.WORK] + self.n_contact_wplace[self.WORK]


    #     #one-hot encoding
    #     df_onehot['type_imune']=sim_mat['type_imune']
    #     df_onehot=pd.get_dummies(df_onehot,columns=['type_imune','age'])
    #     onehot_columns = df_onehot.columns  # 获取所有列的名称
    #     onehot_age_columns = [col for col in onehot_columns if 'age_' in col]
    #     onehot_type_imune_columns = [col for col in onehot_columns if 'type_imune_' in col]
    #
    #     self.stat_individual=df_onehot[onehot_age_columns+onehot_type_imune_columns].to_numpy().astype(np.float32)
    #     #group
    #     wp=df_onehot.groupby(['wplace'])[onehot_age_columns+onehot_type_imune_columns].transform('mean')
    #     wp.iloc[df_onehot['wplace']==-1]=0
    #     wp_num=df_onehot.groupby(['wplace'])['pid'].transform('count')
    #     wp_num.iloc[df_onehot['wplace'] == -1] = 0
    #     wp,wp_num = wp.to_numpy().astype(np.float32),wp_num.to_numpy().astype(np.float32)
    #     self.stat_wp =np.concatenate((wp,wp_num.reshape(-1,1)),axis=-1)
    #
    #     wg=df_onehot.groupby(['wgroup'])[onehot_age_columns+onehot_type_imune_columns].transform('mean')
    #     wg.iloc[df_onehot['wplace']==-1]=0
    #     wg_num=df_onehot.groupby(['wgroup'])['pid'].transform('count')
    #     wg_num.iloc[df_onehot['wplace']==-1]=0
    #
    #     wg,wg_num = wg.to_numpy().astype(np.float32),wg_num.to_numpy().astype(np.float32)
    #
    #     self.stat_wg =np.concatenate((wg,wg_num.reshape(-1,1)),axis=-1)
    #     home=df_onehot.groupby(['hid'])[onehot_age_columns+onehot_type_imune_columns].transform('mean')
    #     home_num=df_onehot.groupby(['hid'])['pid'].transform('count').to_numpy()
    #
    #     home = home.to_numpy().astype(np.float32)
    #     self.stat_home = np.concatenate((home,home_num.reshape(-1,1)),axis=-1)
    #     com=df_onehot.groupby(['hzone'])[onehot_age_columns+onehot_type_imune_columns].transform('mean')
    #     com_num = df_onehot.groupby(['hzone'])['pid'].transform('count').to_numpy()
    #     com = com.to_numpy().astype(np.float32)
    #     self.stat_com = np.concatenate((com,com_num.reshape(-1,1)),axis=-1)
    def get_situation_features(self,sim_mat,sim_date):
        #get region inside risk features
        sim_mat_new=sim_mat.copy()
        if sim_date==0:
            sim_mat_new['comp_this'][(sim_mat['comp_this']<=3)] = 0
            sim_mat_new['comp_this'][(sim_mat['comp_this'] > 3) & (sim_mat['comp_this'] < 7)] = 1
            # sim_mat_new['comp_this'][(sim_mat_new['comp_this'] == 0) ] = 0
            # sim_mat_new['comp_this'][(sim_mat_new['comp_this'] >= 1) & (sim_mat_new['comp_this'] <= 6)] = 1
            sim_mat_new['comp_this'][(sim_mat['comp_this'] >= 7) & (sim_mat['comp_this'] <= 12)] = 2
            self.hea_prob = np.ones((self.n_agent))
            self.com_lag = sim_mat['comp_this']

            self.wg_symp_num =np.zeros((self.n_agent,5))
            self.wp_symp_num =np.zeros((self.n_agent,5))
            self.home_symp_num =np.zeros((self.n_agent,5))
            self.com_symp_num =np.zeros((self.n_agent,5))

            self.intervention = np.zeros((self.n_agent, 5))

        if sim_date>0:
            sim_mat_new['comp_this'][(sim_mat['comp_this'] <= 3)] = 0
            sim_mat_new['comp_this'][(sim_mat['comp_this'] > 3) & (sim_mat['comp_this'] < 7)] = 1
            # sim_mat_new['comp_this'][(sim_mat_new['comp_this'] == 0) ] = 0
            # sim_mat_new['comp_this'][(sim_mat_new['comp_this'] >= 1) & (sim_mat_new['comp_this'] <= 6)] = 1
            sim_mat_new['comp_this'][(sim_mat['comp_this'] >= 7) & (sim_mat['comp_this'] <= 12)] = 2
            sim_mat_new['comp_this'][(self.com_lag==3)&(sim_mat_new['comp_this']==2)]=0#无症状康复不可观测
        # sim_mat_new['type_quran'][sim_mat_new['type_quran'] ==3] = 2
        comp_init_onehot=np.eye(3)[sim_mat_new['comp_this']]
        # print(sim_mat['comp_this'][0])
        # print(comp_init_onehot[0])
        type_quran_onehot=np.eye(5)[sim_mat_new['type_quran']]
        individual_dynamic=np.c_[comp_init_onehot,type_quran_onehot]
        df_info=self.tab_person_rank
        onehot_comp_columns = [f'comp_{i}' for i in range(3)]
        onehot_type_quran_columns =[f'quran_{i}' for i in range(5)]
        comp_df=pd.DataFrame(comp_init_onehot,columns=onehot_comp_columns)
        type_quran_df=pd.DataFrame(type_quran_onehot,columns=onehot_type_quran_columns)
        dynamic_features_df=pd.concat([df_info,comp_df,type_quran_df],axis=1)

        wp_symp_num=dynamic_features_df.groupby('wplace')['comp_1'].transform('sum')
        wp_symp_num=wp_symp_num.to_numpy()
        wg_symp_num=dynamic_features_df.groupby('wgroup')['comp_1'].transform('sum')
        wg_symp_num=wg_symp_num.to_numpy()
        home_symp_num=dynamic_features_df.groupby('hid')['comp_1'].transform('sum')
        home_symp_num=home_symp_num.to_numpy()
        com_symp_num=dynamic_features_df.groupby('hzone')['comp_1'].transform('sum')
        com_symp_num=com_symp_num.to_numpy()

        self.wg_symp_num[:,sim_date%5]=wg_symp_num
        self.wp_symp_num[:,sim_date%5]=wp_symp_num
        self.home_symp_num[:,sim_date%5]=home_symp_num
        self.com_symp_num[:,sim_date%5]=com_symp_num

        self.intervention[:,sim_date%5]=sim_mat_new['type_quran']

        ratio_acq=(np.sum(self.wg_symp_num,axis=1)+np.sum(self.home_symp_num,axis=1)).reshape(-1)/(self.wg_num+self.home_num).reshape(-1)
        prob=np.c_[wg_symp_num/self.wg_num.reshape(-1),wp_symp_num/self.wp_num.reshape(-1),home_symp_num/self.home_num.reshape(-1),com_symp_num/self.com_num.reshape(-1)]
        if sim_date>4:
            inter=np.sum(self.intervention>0,axis=1)
        else:
            if sim_date>0:
                inter=np.sum(self.intervention[:,:sim_date]>0,axis=1)
            else :
                inter=np.zeros((self.n_agent))


        state_infect=sim_mat_new['comp_this']

        time_ratio=((sim_date+1)/self.max_iterday)*np.ones((self.n_agent,1))
        infect_fraction=(np.sum(state_infect==1)/self.n_agent)*np.ones((self.n_agent,1))

        increase_infectons = (np.sum(((self.com_lag <= 3) & (sim_mat['comp_this'] > 3)))/self.n_agent) * np.ones((self.n_agent, 1))
        self.com_lag = sim_mat_new['comp_this']

        frac_iso=(np.sum(sim_mat_new['type_quran']==2)/self.n_agent)*np.ones((self.n_agent,1))
        frac_recovery=(np.sum(sim_mat_new['comp_this']==2)/self.n_agent)*np.ones((self.n_agent,1))

        individual_state=np.c_[ratio_acq,prob,inter,time_ratio,infect_fraction,increase_infectons,frac_iso,frac_recovery]

        region_controlled = dynamic_features_df.groupby('ranked_hzone')[onehot_type_quran_columns].sum().to_numpy()
        region_state = np.zeros((region_controlled.shape[0], 5))
        self.infect_prob=np.zeros(self.n_agent)
        return individual_state,region_state,self.infect_prob, region_controlled
        # wp=dynamic_features_df.groupby('wplace')[onehot_comp_columns].transform('mean')
        # wp.iloc[dynamic_features_df['wplace']==-1]=0
        # wp_inside = wp.to_numpy().astype(np.float32)
        # wg=dynamic_features_df.groupby('wgroup')[onehot_comp_columns].transform('mean')
        # wg.iloc[dynamic_features_df['wplace']==-1]=0
        # wg_inside = wg.to_numpy().astype(np.float32)
        # home=dynamic_features_df.groupby('hid')[onehot_comp_columns].transform('mean')
        # home_inside = home.to_numpy().astype(np.float32)
        # com=dynamic_features_df.groupby('hzone')[onehot_comp_columns].transform('mean')
        # com_inside = com.to_numpy().astype(np.float32)
        #
        # #get region outside risk features
        # # infectious_index = (sim_mat['comp_this'] >3) & (sim_mat['comp_this'] <= 6)
        # wp_symp_num=dynamic_features_df.groupby('wplace')['comp_1'].transform('sum')
        # wp_symp_num=wp_symp_num.to_numpy()
        # wg_symp_num=dynamic_features_df.groupby('wgroup')['comp_1'].transform('sum')
        # wg_symp_num=wg_symp_num.to_numpy()
        # home_symp_num=dynamic_features_df.groupby('hid')['comp_1'].transform('sum')
        # home_symp_num=home_symp_num.to_numpy()
        # com_symp_num=dynamic_features_df.groupby('hzone')['comp_1'].transform('sum')
        # com_symp_num=com_symp_num.to_numpy()
        # inside_num=np.c_[wp_symp_num,wg_symp_num,home_symp_num,com_symp_num]
        # wp_outside_risk,wg_outside_risk,home_outside_risk,com_outside_risk=prob_infectious_model(num_infectious=inside_num,ptran=self.ptrans,region = 'single')
        #
        #
        # dynamic_features_df['wp_outside']=wp_outside_risk
        # dynamic_features_df['wg_outside']=wg_outside_risk
        # dynamic_features_df['home_outside']=home_outside_risk
        # dynamic_features_df['com_outside']=com_outside_risk
        #
        # dynamic_features_df.loc[sim_mat['type_quran']<=self.Q_ISO_HOME,'wp_outside']=0
        # dynamic_features_df.loc[sim_mat['type_quran']<=self.Q_ISO_HOME,'wg_outside']=0
        # home_outside=np.c_[
        #     dynamic_features_df.groupby('hid')['wp_outside'].transform('mean').to_numpy(),
        #     dynamic_features_df.groupby('hid')['wg_outside'].transform('mean').to_numpy()]
        #
        # dynamic_features_df.loc[sim_mat['type_quran'] != self.Q_FREE, 'wp_outside'] = 0
        # dynamic_features_df.loc[sim_mat['type_quran'] != self.Q_FREE, 'wg_outside'] = 0
        # dynamic_features_df.loc[sim_mat['type_quran'] != self.Q_FREE, 'home_outside'] = 0
        # dynamic_features_df.loc[sim_mat['type_quran'] != self.Q_FREE, 'com_outside'] = 0
        #
        # com_outside=np.c_[
        #     dynamic_features_df.groupby('hzone')['wp_outside'].transform('mean').to_numpy(),
        #     dynamic_features_df.groupby('hzone')['wg_outside'].transform('mean').to_numpy()]
        #
        # wp_outside=np.c_[
        #     dynamic_features_df.groupby('wplace')['home_outside'].transform('mean').to_numpy(),
        #     dynamic_features_df.groupby('wplace')['com_outside'].transform('mean').to_numpy()]
        #
        # wg_outside = np.c_[
        #     dynamic_features_df.groupby('wgroup')['home_outside'].transform('mean').to_numpy(),
        #     dynamic_features_df.groupby('wgroup')['com_outside'].transform('mean').to_numpy()]
        #
        #
        #
        #
        # if sim_date==0:
        #     self.increase_infectons_3day=deque(maxlen=3)
        #     increase_infectons=np.c_[wp_inside[:,0],wg_inside[:,0],home_inside[:,0],com_inside[:,0]]
        #     self.increase_infectons_3day.append(increase_infectons)
        #     deta3=np.zeros((self.n_agent,4))
        #     self.increase_infectons_5day = deque(maxlen = 3)
        #     increase_infectons = np.c_[wp_inside[:, 0], wg_inside[:, 0], home_inside[:, 0], com_inside[:, 0]]
        #     self.increase_infectons_5day.append(increase_infectons)
        #     deta5 = np.zeros((self.n_agent, 4))
        #     self.features = np.c_[
        #         self.stat_individual, self.stat_wp, self.stat_wg, self.stat_home, self.stat_com,individual_dynamic,
        #         wp_inside,wp_outside, wg_inside,wg_outside, home_inside, home_outside, com_inside,com_outside, deta3, deta5]
        #
        # if sim_date>0:
        #     self.increase_infectons_3day.append(np.c_[wp_inside[:, 0], wg_inside[:, 0], home_inside[:, 0], com_inside[:, 0]])
        #     deta3 = (np.array(self.increase_infectons_3day[0]) - np.array(self.increase_infectons_3day[-1]))
        #     self.increase_infectons_5day.append(np.c_[wp_inside[:,0],wg_inside[:,0],home_inside[:,0],com_inside[:,0]])
        #     deta5=(np.array(self.increase_infectons_5day[0])-np.array(self.increase_infectons_5day[-1]))
        #     self.features[:,74:81]=individual_dynamic
        #     self.features[sim_mat['type_quran']==self.Q_FREE,81:]= np.c_[ wp_inside,wp_outside, wg_inside,wg_outside, home_inside, home_outside, com_inside,com_outside, deta3, deta5][sim_mat['type_quran']==self.Q_FREE,:]
        #     self.features[sim_mat['type_quran']<=self.Q_ISO_HOME,91:96]= np.c_[home_inside,home_outside][sim_mat['type_quran']<=self.Q_ISO_HOME,:]
        #     self.features[sim_mat['type_quran']<=self.Q_ISO_HOME,103]= deta3[sim_mat['type_quran']<=self.Q_ISO_HOME,2]
        #     self.features[sim_mat['type_quran']<=self.Q_ISO_HOME,107]= deta5[sim_mat['type_quran']<=self.Q_ISO_HOME,2]

        # get infectious num
        # infectious_index=(sim_mat['comp_this']>3) & (sim_mat['comp_this']<=6)
        # infectious_index = (sim_mat['comp_this'] > 3)
        # dynamic_features_df['num_asym_wp']=infectious_index
        # dynamic_features_df['num_asym_wg']=infectious_index
        # dynamic_features_df['num_asym_home']=infectious_index
        # dynamic_features_df['num_asym_com']=infectious_index
        # num_wp=dynamic_features_df.groupby('wplace')['num_asym_wp'].transform('sum')
        # num_wg=dynamic_features_df.groupby('wgroup')['num_asym_wg'].transform('sum')
        # num_home=dynamic_features_df.groupby('hid')['num_asym_home'].transform('sum')
        # num_com=dynamic_features_df.groupby('hzone')['num_asym_com'].transform('sum')

        # dynamic_features_df['num_asym'] = infectious_index
        # # dynamic_features_df['num_asym_wg']=(sim_mat['type_quran'] == self.Q_FREE) & infectious_index
        # # dynamic_features_df['num_asym_home']=(sim_mat['type_quran'] == self.Q_FREE) & infectious_index
        # # dynamic_features_df['num_asym_isohome']=(sim_mat['type_quran'] == self.Q_ISO_HOME) & infectious_index
        # # dynamic_features_df['num_asym_com']=(sim_mat['type_quran'] == self.Q_FREE) & infectious_index
        # num_wp=dynamic_features_df.groupby('wplace')['num_asym'].transform('sum')
        # num_wg=dynamic_features_df.groupby('wgroup')['num_asym'].transform('sum')
        # num_home=dynamic_features_df.groupby('hid')['num_asym'].transform('sum')
        # num_isohome=dynamic_features_df.groupby('hid')['num_asym_iso'].transform('sum')
        # num_com=dynamic_features_df.groupby('hzone')['num_asym'].transform('sum')
        #
        # if sim_date==0:

        # prob_model_features=np.c_[num_wp,num_wg,num_home,num_com]#[4]
        # # if sim_date>0:
        # #     self.prob_model_features[sim_mat['type_quran']==self.Q_FREE,0]=num_wp[sim_mat['type_quran']==self.Q_FREE]
        # #     self.prob_model_features[sim_mat['type_quran']==self.Q_FREE,1]=num_wg[sim_mat['type_quran']==self.Q_FREE]
        # #
        # #     self.prob_model_features[sim_mat['type_quran']==self.Q_FREE,2]=num_home[sim_mat['type_quran']==self.Q_FREE]
        # #     self.prob_model_features[sim_mat['type_quran']==self.Q_FREE,3]=num_com[sim_mat['type_quran']==self.Q_FREE]
        #
        # prob_infecious_base,prob_non_base=prob_infectious_model(num_infectious=prob_model_features,ptran=self.ptrans)
        #
        # #get by asym
        # dynamic_features_df['prob_base'] =prob_infecious_base
        #
        # num_wp_asym = dynamic_features_df.groupby('wplace')['prob_base'].transform('sum')
        # num_wg_asym  = dynamic_features_df.groupby('wgroup')['prob_base'].transform('sum')
        # num_home_asym  = dynamic_features_df.groupby('hid')['prob_base'].transform('sum')
        # num_com_asym  = dynamic_features_df.groupby('hzone')['prob_base'].transform('sum')
        # prob_model_features_asym = np.c_[num_wp_asym, num_wg_asym, num_home_asym, num_com_asym]  # [4]
        # prob_infecious_asym, prob_non_asym = prob_infectious_model(num_infectious=prob_model_features_asym, ptran=self.ptrans)
        #
        # self.prob_infecious=1-(prob_non_base)*(prob_non_asym)

        # if sim_date==0:
        #     self.increase_num=np.zeros((self.n_agent,5,7))
        #     self.prob_infecious=np.zeros((self.n_agent))
        #     self.com_lag=sim_mat['comp_this']
        #     self.control_state=np.zeros((4,self.n_agent,7))
        #
        # elif sim_date>0:
        #
        #     self.control_state[0, :, sim_date % 7] = np.array(sim_mat['type_quran'] == self.Q_FREE).astype(float)
        #     self.control_state[1, :, sim_date % 7] = np.array(sim_mat['type_quran'] == self.Q_COM).astype(float)
        #     self.control_state[2, :, sim_date % 7] = np.array(sim_mat['type_quran'] == self.Q_ISO_HOME).astype(float)
        #     self.control_state[3, :, sim_date % 7] = np.array(
        #         (sim_mat['type_quran'] == self.Q_ISO) | (sim_mat['type_quran'] == self.Q_HOSP)).astype(float)
        #     if sim_date-3<=0:
        #         if_out_com=np.array(np.sum(self.control_state[0, :, 1:sim_date+1],axis = 1)>0).astype(float)
        #         if_com=np.array((if_out_com+np.array(np.sum(self.control_state[1, :, 1:sim_date+1],axis = 1)>0).astype(float))>0).astype(float)
        #         if_home=np.array((if_com+np.array(np.sum(self.control_state[2, :, 1:sim_date+1],axis = 1)>0).astype(float))>0).astype(float)
        #
        #     elif sim_date-3>0:
        #         if_out_com = np.array(np.sum(self.control_state[0, :, sim_date % 7-2:sim_date % 7+1], axis=1) > 0).astype(float)
        #         if_com = np.array((if_out_com + np.array(np.sum(self.control_state[1, :, sim_date % 7-2:sim_date % 7+1], axis=1) > 0).astype(float))>0).astype(float)
        #         if_home = np.array((if_com + np.array(np.sum(self.control_state[2, :, sim_date % 7-2:sim_date % 7+1], axis=1) > 0).astype(float))>0).astype(float)
        #
        #     increase_infectons = ((self.com_lag<=3) & (sim_mat['comp_this']>3)).astype(float)
        #     dynamic_features_df['increase_infectons_out_com']=np.array(increase_infectons)*if_out_com
        #     dynamic_features_df['increase_infectons_com'] = np.array(increase_infectons) * if_com
        #     dynamic_features_df['increase_infectons_home'] = np.array(increase_infectons) * if_home
        #
        #     increase_wp = dynamic_features_df.groupby('wplace')['increase_infectons_out_com'].transform('sum').to_numpy()
        #     increase_wp[self.tab_person['wtype']==-1]=0
        #     increase_wp=increase_wp*if_out_com
        #
        #     increase_wg = dynamic_features_df.groupby('wgroup')['increase_infectons_out_com'].transform('sum').to_numpy()
        #     increase_wg[self.tab_person['wtype'] == -1] = 0
        #     increase_wg = increase_wg * if_out_com
        #     increase_jd = dynamic_features_df.groupby('h_jd')['increase_infectons_out_com'].transform(
        #         'sum').to_numpy()
        #     increase_jd=increase_jd*if_com
        #
        #
        #     increase_home = dynamic_features_df.groupby('hid')['increase_infectons_home'].transform('sum').to_numpy()
        #     increase_home= increase_home * if_home
        #
        #     increase_com = dynamic_features_df.groupby('hzone')['increase_infectons_com'].transform('sum').to_numpy()
        #     increase_com = increase_com * if_com
        #
        #     increase_num=np.c_[increase_wp,increase_wg,increase_home,increase_com,increase_jd]
        #     self.increase_num[:,:,sim_date%7]=increase_num
        #
        #     if sim_date<7:
        #         self.prob_infecious=1-np.prod(1-(self.increase_num[:,0,:sim_date+1]/self.wp_num),axis=1)\
        #                             *np.prod(1-(self.increase_num[:,1,:sim_date+1]/self.wg_num),axis=1)\
        #                             *np.prod(1-(self.increase_num[:,2,:sim_date+1]/self.home_num),axis=1)\
        #                             *np.prod(1-(self.increase_num[:,3,:sim_date+1]/self.com_num),axis=1)
        #
        #     if sim_date>=7:
        #         self.prob_infecious=1-np.prod(1-(self.increase_num[:,0,:]/self.wp_num),axis=1)\
        #                             *np.prod(1-(self.increase_num[:,1,:]/self.wg_num),axis=1)\
        #                             *np.prod(1-(self.increase_num[:,2,:]/self.home_num),axis=1)\
        #                             *np.prod(1-(self.increase_num[:,3,:]/self.com_num),axis=1)
        #     if sim_date < 7:
        #         self.prob_infecious1 = 1 - (
        #                     1 - np.mean(self.increase_num[:, 0, :sim_date], axis = 1) / self.wp_num.reshape(-1)) * (
        #                                           1 - np.mean(self.increase_num[:, 1, :sim_date],
        #                                                       axis = 1) / self.wg_num.reshape(-1)) * (
        #                                           1 - np.mean(self.increase_num[:, 2, :sim_date],
        #                                                       axis = 1) / self.home_num.reshape(-1)) * (
        #                                           1 - np.mean(self.increase_num[:, 3, :sim_date],
        #                                                       axis = 1) / self.com_num.reshape(-1))
        #     if sim_date >= 7:
        #         self.prob_infecious1 = 1 - (1 - np.mean(self.increase_num[:, 0, :], axis = 1) / self.wp_num.reshape(-1)) * (
        #                 1 - np.mean(self.increase_num[:, 1, :], axis = 1) / self.wg_num.reshape(-1)) * (
        #                                       1 - np.mean(self.increase_num[:, 2, :], axis = 1) / self.home_num.reshape(-1)) * (
        #                                       1 - np.mean(self.increase_num[:, 3, :], axis = 1) / self.com_num.reshape(-1))
        #
        #         # self.prob_infecious = 1 - (1 - self.increase_num[:, 0, -1] / self.wp_num) * (
        #         #         1 -self.increase_num[:, 1, -1] / self.wg_num) * (
        #         #                               1 - self.increase_num[:, 2, -1] / self.home_num) * (
        #         #                               1 - self.increase_num[:, 3, -1] / self.com_num)
        #     self.com_lag = sim_mat['comp_this']
        #
        # #add a random noise on prob_infecious use numpy
        # # self.prob_infecious=np.minimum(self.prob_infecious+np.abs(np.random.normal(0,0.1,self.prob_infecious.shape)),1)
        # #get community feature
        # self.prob_infecious[np.where(sim_mat['comp_this'] >= 4)[0]] = 0
        # dynamic_features_df['prob>0']=(self.prob_infecious>0).astype(float)
        # dynamic_features_df['prob>0.1'] = (self.prob_infecious > 0.1).astype(float)
        # prob_column=['prob>0','prob>0.1']
        # com_features=dynamic_features_df.groupby('ranked_hzone')[onehot_comp_columns+onehot_type_quran_columns+prob_column].sum()
        # # com_symp_num = dynamic_features_df.groupby('ranked_hzone')[
        # #     onehot_comp_columns].mean()
        # com_features=com_features.to_numpy()
        # global_features=np.zeros_like(com_features)
        # interm_features=np.zeros_like(com_features)
        # global_features[0,:]=dynamic_features_df[onehot_comp_columns+onehot_type_quran_columns+prob_column].sum().to_numpy()
        # interm_features[:dynamic_features_df['ranked_jdzone'].max()+1,:]=dynamic_features_df.groupby('ranked_jdzone')[onehot_comp_columns+onehot_type_quran_columns+prob_column].sum().to_numpy()
        # com_num = dynamic_features_df.groupby(['ranked_hzone'])['pid'].count().to_numpy()
        # com_num=com_num/np.max(com_num)
        # self.features=np.c_[com_num,com_symp_num]



        # self.features=np.c_[global_features,com_features]
        #transform self.stat_prob into 0-1
        # self.stat_prob=(self.stat_prob-np.min(self.stat_prob))/(np.max(self.stat_prob)-np.min(self.stat_prob))
        # self.stat_prob=self.stat_prob*(np.max(self.prob_infecious))
        #
        # # self.prob_infecious = None
        #
        # prob=(self.stat_prob,self.prob_infecious)
        # dynamic_features_df['infected'] = (sim_mat['comp_this'] > 0).astype(float)
        # region_infected = dynamic_features_df.groupby('ranked_hzone')['infected'].sum().to_numpy()
        # region_controlled=dynamic_features_df.groupby('ranked_hzone')[onehot_type_quran_columns].sum().to_numpy()
        # return self.features,prob,region_infected,region_controlled


    def simulate_vaccination(self, sim_mat):
        if self.uptake_scenario=="Uptake Current":
            """ 接种率采用2022年10月CCDC提供的接种率数据 效力衰减到基础值 """
            tab_uptake_age = self._get_vaccination_coverage()
            for age_code,age_group in enumerate(tab_uptake_age.columns):
                pid_this_age = np.random.permutation(np.where(self.tab_person["age"]==age_code)[0])
                n_cums = (tab_uptake_age[age_group].values.cumsum()[:-1]*len(pid_this_age)).astype(int)
                pid_0dose,pid_1dose,pid_2dose,pid_3dose = np.split(pid_this_age,n_cums)
                sim_mat["type_imune"][pid_0dose] = self.V0
                sim_mat["type_imune"][pid_1dose] = self.V1
                sim_mat["type_imune"][pid_2dose] = self.V2
                sim_mat["type_imune"][pid_3dose] = self.V3
                sim_mat["days_imune"] = 180

        elif self.uptake_scenario=="Uptake Enhence":
            """ 在2022年10月CCDC提供的接种率数据 的基础上再补种一针 效力衰减到0-60天 """
            tab_uptake_age = self._get_vaccination_coverage()
            for age_code,age_group in enumerate(tab_uptake_age.columns):
                pid_this_age = np.random.permutation(np.where(self.tab_person["age"]==age_code)[0])
                n_cums = (tab_uptake_age[age_group].values.cumsum()[:-1]*len(pid_this_age)).astype(int)
                pid_0dose,pid_1dose,pid_2dose,pid_3dose = np.split(pid_this_age,n_cums)
                if age_code==0:                                                             #  <----------------- 0-2岁不参与补种
                    sim_mat["type_imune"][pid_0dose] = self.V0
                    sim_mat["type_imune"][pid_1dose] = self.V1
                    sim_mat["type_imune"][pid_2dose] = self.V2
                    sim_mat["type_imune"][pid_3dose] = self.V3
                else:
                    sim_mat["type_imune"][pid_0dose] = self.V1
                    sim_mat["type_imune"][pid_1dose] = self.V2
                    sim_mat["type_imune"][pid_2dose] = self.V3
                    sim_mat["type_imune"][pid_3dose] = self.V3
            sim_mat["days_imune"] = np.random.uniform(0,60,self.n_agent)

        elif self.uptake_scenario=="Uptake Enhence 60":
            """ 在2022年10月CCDC提供的接种率数据 的基础上 60+ 再补种一针 效力衰减到0-60天 """
            tab_uptake_age = self._get_vaccination_coverage()
            for age_code,age_group in enumerate(tab_uptake_age.columns):
                pid_this_age = np.random.permutation(np.where(self.tab_person["age"]==age_code)[0])
                n_cums = (tab_uptake_age[age_group].values.cumsum()[:-1]*len(pid_this_age)).astype(int)
                pid_0dose,pid_1dose,pid_2dose,pid_3dose = np.split(pid_this_age,n_cums)
                if age_code<10:                                                             #  <----------------- 60岁以下不参与补种
                    sim_mat["type_imune"][pid_0dose] = self.V0
                    sim_mat["type_imune"][pid_1dose] = self.V1
                    sim_mat["type_imune"][pid_2dose] = self.V2
                    sim_mat["type_imune"][pid_3dose] = self.V3
                else:
                    sim_mat["type_imune"][pid_0dose] = self.V1
                    sim_mat["type_imune"][pid_1dose] = self.V2
                    sim_mat["type_imune"][pid_2dose] = self.V3
                    sim_mat["type_imune"][pid_3dose] = self.V3
            sim_mat["days_imune"] = np.random.uniform(0,60,self.n_agent)


        elif self.uptake_scenario=="Uptake Enhence Homo":
            """ 在2022年10月CCDC提供的接种率数据 的基础上再补种一针 效力衰减到0-60天 """
            tab_uptake_age = self._get_vaccination_coverage()
            for age_code,age_group in enumerate(tab_uptake_age.columns):
                pid_this_age = np.random.permutation(np.where(self.tab_person["age"]==age_code)[0])
                n_cums = (tab_uptake_age[age_group].values.cumsum()[:-1]*len(pid_this_age)).astype(int)
                pid_0dose,pid_1dose,pid_2dose,pid_3dose = np.split(pid_this_age,n_cums)
                if age_code==0:                                                             #  <----------------- 0-2岁不参与补种
                    sim_mat["type_imune"][pid_0dose] = self.V0
                    sim_mat["type_imune"][pid_1dose] = self.V1
                    sim_mat["type_imune"][pid_2dose] = self.V2
                    sim_mat["type_imune"][pid_3dose] = self.V3
                else:
                    sim_mat["type_imune"][pid_0dose] = self.V1
                    sim_mat["type_imune"][pid_1dose] = self.V2
                    sim_mat["type_imune"][pid_2dose] = self.V3
                    sim_mat["type_imune"][pid_3dose] = self.V3
            sim_mat["days_imune"] = np.random.uniform(0,180,self.n_agent)

        elif self.uptake_scenario=="Uptake Enhence 60 Homo":
            """ 在2022年10月CCDC提供的接种率数据 的基础上 60+ 再补种一针 效力衰减到0-60天 """
            tab_uptake_age = self._get_vaccination_coverage()
            for age_code,age_group in enumerate(tab_uptake_age.columns):
                pid_this_age = np.random.permutation(np.where(self.tab_person["age"]==age_code)[0])
                n_cums = (tab_uptake_age[age_group].values.cumsum()[:-1]*len(pid_this_age)).astype(int)
                pid_0dose,pid_1dose,pid_2dose,pid_3dose = np.split(pid_this_age,n_cums)
                if age_code<10:                                                             #  <----------------- 60岁以下不参与补种
                    sim_mat["type_imune"][pid_0dose] = self.V0
                    sim_mat["type_imune"][pid_1dose] = self.V1
                    sim_mat["type_imune"][pid_2dose] = self.V2
                    sim_mat["type_imune"][pid_3dose] = self.V3
                else:
                    sim_mat["type_imune"][pid_0dose] = self.V1
                    sim_mat["type_imune"][pid_1dose] = self.V2
                    sim_mat["type_imune"][pid_2dose] = self.V3
                    sim_mat["type_imune"][pid_3dose] = self.V3
            sim_mat["days_imune"] = np.random.uniform(0,180,self.n_agent)

        elif self.uptake_scenario=="Uptake 90":
            pid_adult = np.random.permutation(self.tab_person["pid"][self.tab_person["age"]>=5])
            n_cums = [int(0.1*len(pid_adult))]
            pid_2dose, pid_3dose = np.split(pid_adult, n_cums)
            sim_mat["type_imune"][pid_2dose] = self.V2
            sim_mat["type_imune"][pid_3dose] = self.V3

            pid_teen = self.tab_person["pid"][(1<=self.tab_person["age"])*(self.tab_person["age"]<=4)]
            sim_mat["type_imune"][pid_teen ] = self.V2

            pid_child = self.tab_person["pid"][self.tab_person["age"]==0]
            sim_mat["type_imune"][pid_child] = self.V0

        elif self.uptake_scenario=="Uptake 00":
            sim_mat["type_imune"] = self.V0
        else:
            raise ValueError("Wrong Input of Uptake Scenario!")
        # sim_mat[-1]["type_imune"]=self.V0
        # sim_mat[-1]["days_imune"]=0
        return sim_mat

    def simulate_drug_access(self, sim_mat):
        """
        确定药物供给的人群
        按照Mortality Risk和年龄进行药物供给时, 要对同年龄组的个体进行Randomize以避免引入bias
        Randomize numpy.argsort output in case of ties: https://stackoverflow.com/questions/56752412
        """
        pid_unprotected = self.tab_person[(sim_mat["comp_this"]!=self.PROT)&(self.tab_person["age"]>=10)]["pid"]
        age_unprotected = self.tab_person["age"][pid_unprotected]

        if self.priority_drug=="Mortality":
            motality_risk = self._get_age_specified_motality_risk()
            order_age = motality_risk.argsort().argsort()
            rank_drug = (order_age[age_unprotected]+np.random.rand(len(age_unprotected))).argsort()[::-1]

        elif self.priority_drug=="Eldly":
            rank_drug = (age_unprotected+np.random.rand(len(age_unprotected))).argsort()[::-1]

        elif self.priority_drug=="Random":
            rank_drug = np.random.permutation(age_unprotected)

        n_p_drug = (np.round(self.p_drug*len(age_unprotected))).astype(int)
        pid_drug = pid_unprotected[rank_drug[:n_p_drug]]
        sim_mat["ac_drug"][pid_drug] = True
        return sim_mat

    def make_duration_lognorm(self, transition, size):
        scale = self.compart_duration[transition]
        durations = lognorm.rvs(scale=scale, s=self.duration_sigma, size=size)
        return durations

    def select_pid_from_zone(self, sqcode):
        """指定社区中选取pid"""
        pid_cand = self.tab_person[self.tab_person["w_sq"]==sqcode]["pid"]
        pid_seed = np.random.choice(pid_cand)
        return pid_seed

    def get_pid_imported_daily_by_list(self, sim_date):
        if len(self.list_imported_daily):
            sqcodes = self.list_imported_daily[self.list_imported_daily[:, 0]==sim_date, 1]
            pid_imported_daily = np.fromiter(map(lambda x: self.select_pid_from_zone(x), sqcodes), dtype=int)
        else:
            pid_imported_daily = np.empty(shape=0, dtype=int)
        return pid_imported_daily

    def simulate_imported(self, sim_mat, sim_date):
        if sim_date==0:                                                                                            # 第0天 投入n_imported_day0数量随机种子 或种子列表
            if self.n_imported_day0>0:
                pid_imported_daily = np.random.choice(self.pid_unprotect, self.n_imported_day0, replace=False)     # 采用随机种子
            else:
                pid_imported_daily = self.get_pid_imported_daily_by_list(sim_date)                                 # 采用种子列表
        else:                                                                                                      # 之后每天 投入n_imported_daily数量随机种子 或种子列表
            pid_imported_daily_list = self.get_pid_imported_daily_by_list(sim_date)
            pid_imported_daily_rand = np.random.choice(self.pid_unprotect, self.n_imported_daily)                  # 每日感染种子不会出现在养老院
            pid_imported_daily = np.r_[pid_imported_daily_list,pid_imported_daily_rand]
            pid_imported_daily = pid_imported_daily[sim_mat["comp_this"][pid_imported_daily]==self.SUSC]

        if pid_imported_daily.size:                                                                                # 更新 sim_mat
            n_imported = len(pid_imported_daily)
            sim_mat["comp_this"  ][pid_imported_daily] = np.repeat(self.LATT, n_imported)
            sim_mat["comp_next"  ][pid_imported_daily] = np.repeat(self.INCB, n_imported)
            sim_mat["cd_trans"   ][pid_imported_daily] = self.make_duration_lognorm("latt2incb", n_imported)
            sim_mat["days_infect"][pid_imported_daily] = 0

            date_pid_imported_daily = np.c_[np.repeat(sim_date, n_imported), pid_imported_daily]                   # 存储到 self.date_pid_imported
            self.date_pid_imported.append(date_pid_imported_daily)
        return sim_mat

    def update_days(self, sim_mat):
        # self.gdf_sq["cd_risk"] -= 1
        sim_mat["cd_trans"   ] -= 1
        sim_mat["cd_quran"   ] -= 1
        sim_mat["days_infect"] += 1
        sim_mat["days_imune" ] += 1
        sim_mat["days_imune" ] = np.clip(sim_mat["days_imune"], 0, self.max_VE_span-1)
        return sim_mat

    def get_pid_with_outcome(self, sim_mat, pids, transition, epsilon=0, index_only=False):
        immune_type = sim_mat[pids]["type_imune"]
        immune_days = sim_mat[pids]["days_imune"]
        target_ages = self.tab_person[pids]["age"]
        p_attrs = np.c_[immune_type, target_ages, immune_days]

        s = np.fromiter(map(lambda x: self.trans_rate[transition].get((x[0], x[1]))[x[2]], p_attrs), dtype=float)
        s = s*(1-epsilon)
        r = np.random.rand(s.shape[0])

        if index_only:
            idx_outcome0 = np.where(~(r>s))[0]
            idx_outcome1 = np.where( (r>s))[0]
            return idx_outcome0, idx_outcome1
        else:
            pid_outcome0 = pids[~(r>s)]
            pid_outcome1 = pids[ (r>s)]
            return pid_outcome0,pid_outcome1

    def simulate_isolation_home_asym(self, sim_mat, pid_asym):
        # if self.is_work_off==False:
        #     return sim_mat  # 未停工asym全都不居家
        # else:
        pid_iso_p = self._random_binomial_choice(pid_asym,  self.rate_iso_p_work_off)
        pid_iso_h = self._random_binomial_choice(pid_iso_p, self.rate_iso_h_work_off)
        hid_iso_h = np.unique(self.tab_person["hid"][pid_iso_h])
        pid_iso_h = np.concatenate(list(map(self.hdict.get, hid_iso_h))) if hid_iso_h.size else np.empty(shape=0, dtype=int)

        pid_iso_h = pid_iso_h[sim_mat["type_quran"][pid_iso_h]==self.Q_FREE]
        pid_iso_p = pid_iso_p[sim_mat["type_quran"][pid_iso_p]==self.Q_FREE]

        # sim_mat["type_quran"][pid_iso_h] = self.Q_ISO_H
        sim_mat["type_quran"][pid_iso_p] = self.Q_ISO_HOME
        sim_mat["cd_quran"  ][pid_iso_h] = self.days_iso_h
        sim_mat["cd_quran"  ][pid_iso_p] = self.days_iso_p
        return sim_mat

    def simulate_isolation_home_mild(self, sim_mat, pid_mild):
        # if self.is_work_off==False:
        #     return sim_mat  # 未停工mild全都不居家
        # else:
        pid_iso_p = self._random_binomial_choice(pid_mild,  self.rate_iso_p_work_off)
        pid_iso_h = self._random_binomial_choice(pid_iso_p, self.rate_iso_h_work_off)
        hid_iso_h = np.unique(self.tab_person["hid"][pid_iso_h])
        pid_iso_h = np.concatenate(list(map(self.hdict.get, hid_iso_h))) if hid_iso_h.size else np.empty(shape=0, dtype=int)

        pid_iso_h = pid_iso_h[sim_mat["type_quran"][pid_iso_h]==self.Q_FREE]
        pid_iso_p = pid_iso_p[sim_mat["type_quran"][pid_iso_p]==self.Q_FREE]

        # sim_mat["type_quran"][pid_iso_h] = self.Q_ISO_H
        sim_mat["type_quran"][pid_iso_p] = self.Q_ISO_HOME
        sim_mat["cd_quran"  ][pid_iso_h] = self.days_iso_h
        sim_mat["cd_quran"  ][pid_iso_p] = self.days_iso_p
        return sim_mat

    def simulate_incb(self, sim_mat, sim_mat_new):
        # if not (pid_incb:=sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.INCB)]["pid"]).size:
        pid_incb = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.INCB)]["pid"]
        if not pid_incb.size:
            return sim_mat_new

        pid_symp, pid_asym = self.get_pid_with_outcome(sim_mat, pid_incb, "incb2symp")
        # update agent current health status
        sim_mat_new["comp_this"][pid_incb] = self.INCB

        # update agent future health status
        sim_mat_new["comp_next"][pid_symp] = self.SYMP
        sim_mat_new["comp_next"][pid_asym] = self.ASYM

        # update period of this health status
        sim_mat_new["cd_trans" ][pid_symp] += self.make_duration_lognorm("incb2symp", len(pid_symp))
        sim_mat_new["cd_trans" ][pid_asym] += self.make_duration_lognorm("incb2asym", len(pid_asym))
        return sim_mat_new

    def simulate_symp(self, sim_mat, sim_mat_new):
        # if not (pid_symp:=sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.SYMP)]["pid"]).size:
        pid_symp = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.SYMP)]["pid"]
        if not pid_symp.size:
            return sim_mat_new

        sim_mat_symp = sim_mat[pid_symp]
        pid_symp_drug    = sim_mat_symp["pid"][ sim_mat_symp["ac_drug"]]
        pid_symp_no_drug = sim_mat_symp["pid"][~sim_mat_symp["ac_drug"]]

        pid_seve_0, pid_mild_0 = self.get_pid_with_outcome(sim_mat, pid_symp_drug,    "symp2seve", epsilon=self.e_drug) if pid_symp_drug.size else [np.empty(shape=0, dtype=int)]*2
        pid_seve_1, pid_mild_1 = self.get_pid_with_outcome(sim_mat, pid_symp_no_drug, "symp2seve", epsilon=0) if pid_symp_no_drug.size else [np.empty(shape=0, dtype=int)]*2
        pid_seve = np.r_[pid_seve_0, pid_seve_1]
        pid_mild = np.r_[pid_mild_0, pid_mild_1]

        sim_mat_new["comp_this"][pid_symp] = self.SYMP
        sim_mat_new["comp_next"][pid_seve] = self.SEVE
        sim_mat_new["comp_next"][pid_mild] = self.MILD
        return sim_mat_new

    def simulate_asym(self, sim_mat, sim_mat_new):
        # if not (pid_asym:=sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.ASYM)]["pid"]).size:
        pid_asym = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.ASYM)]["pid"]
        if not pid_asym.size:
            return sim_mat_new

        sim_mat_new["comp_this"][pid_asym] = self.ASYM
        sim_mat_new["comp_next"][pid_asym] = self.RECV
        sim_mat_new["cd_trans" ][pid_asym] += self.make_duration_lognorm("asym2recv", len(pid_asym))
        sim_mat_new = self.simulate_isolation_home_asym(sim_mat_new, pid_asym)
        return sim_mat_new

    def simulate_mild(self, sim_mat, sim_mat_new):
        pid_mild = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.MILD)]["pid"]
        if not pid_mild.size:
            return sim_mat_new

        sim_mat_new["comp_this"][pid_mild] = self.MILD
        sim_mat_new["comp_next"][pid_mild] = self.RECV
        sim_mat_new["cd_trans" ][pid_mild] += self.make_duration_lognorm("mild2recv", len(pid_mild))
        sim_mat_new = self.simulate_isolation_home_mild(sim_mat_new, pid_mild)
        return sim_mat_new

    def simulate_seve(self, sim_mat, sim_mat_new):
        pid_seve = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.SEVE)]["pid"]
        if not pid_seve.size:
            return sim_mat_new

        pid_icu, pid_hosp = self.get_pid_with_outcome(sim_mat, pid_seve, "seve2icu")
        # sim_mat_new["type_quran"][pid_seve] = self.Q_ISO_HOME
        sim_mat_new["comp_this" ][pid_seve] = self.SEVE
        sim_mat_new["comp_next" ][pid_icu ] = self.ICU
        sim_mat_new["comp_next" ][pid_hosp] = self.HOSP
        sim_mat_new["cd_trans"  ][pid_icu ] += self.make_duration_lognorm("seve2icu",  len(pid_icu))
        sim_mat_new["cd_trans"  ][pid_hosp] += self.make_duration_lognorm("seve2hosp", len(pid_hosp))
        return sim_mat_new

    def simulate_hosp(self, sim_mat, sim_mat_new):
        pid_hosp = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.HOSP)]["pid"]
        if not pid_hosp.size:
            return sim_mat_new

        pid_death, pid_recv = self.get_pid_with_outcome(sim_mat, pid_hosp, "hosp2death", epsilon=self.e_drug_death)
        sim_mat_new["comp_this" ][pid_hosp ] =  self.HOSP
        sim_mat_new["comp_next" ][pid_death] =  self.DEATH
        sim_mat_new["comp_next" ][pid_recv ] =  self.RECV
        sim_mat_new["cd_trans"  ][pid_death] += self.make_duration_lognorm("hosp2death", len(pid_death))
        sim_mat_new["cd_trans"  ][pid_recv ] += self.make_duration_lognorm("hosp2recv",  len(pid_recv ))
        sim_mat_new["type_quran"][pid_hosp ] =  self.Q_HOSP
        sim_mat_new["cd_quran"  ][pid_hosp ] =  np.inf
        return sim_mat_new

    def simulate_icu(self, sim_mat, sim_mat_new):
        pid_icu = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.ICU)]["pid"]
        if not pid_icu.size:
            return sim_mat_new

        pid_death, pid_hospr = self.get_pid_with_outcome(sim_mat, pid_icu, "icu2death", epsilon=self.e_drug_death)
        sim_mat_new["comp_this" ][pid_icu  ] =  self.ICU
        sim_mat_new["comp_next" ][pid_death] =  self.DEATH
        sim_mat_new["comp_next" ][pid_hospr] =  self.HOSPR
        sim_mat_new["cd_trans"  ][pid_death] += self.make_duration_lognorm("icu2death", len(pid_death))
        sim_mat_new["cd_trans"  ][pid_hospr] += self.make_duration_lognorm("icu2hospr", len(pid_hospr))
        sim_mat_new["type_quran"][pid_icu  ] =  self.Q_HOSP
        sim_mat_new["cd_quran"  ][pid_icu  ] =  np.inf
        return sim_mat_new

    def simulate_hospr(self, sim_mat, sim_mat_new):
        pid_hospr = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.HOSPR)]["pid"]
        if not pid_hospr.size:
            return sim_mat_new

        sim_mat_new["comp_this"][pid_hospr] =  self.HOSPR
        sim_mat_new["comp_next"][pid_hospr] =  self.RECV
        sim_mat_new["cd_trans" ][pid_hospr] += self.make_duration_lognorm("hospr2recv", len(pid_hospr))
        return sim_mat_new

    def simulate_recv(self, sim_mat, sim_mat_new):
        pid_recv = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.RECV)]["pid"]
        if not pid_recv.size:
            return sim_mat_new

        sim_mat_new["comp_this"][pid_recv] =  self.RECV
        sim_mat_new["comp_next"][pid_recv] =  self.SUSC
        sim_mat_new["cd_trans" ][pid_recv] += self.make_duration_lognorm("recv2susc", len(pid_recv))

        pid_recv_from_hosp = pid_recv[sim_mat["type_quran"][pid_recv]==self.Q_HOSP]
        if not pid_recv_from_hosp.size:
            return sim_mat_new

        if self.days_iso_discharge>0:
            sim_mat_new["type_quran"][pid_recv_from_hosp] = self.Q_DISCH
            sim_mat_new["cd_quran"  ][pid_recv_from_hosp] = self.days_iso_discharge

        else:
            sim_mat_new["type_quran"][pid_recv_from_hosp] = self.Q_FREE
            # print("recovery id:",sim_mat_new["pid"][pid_recv_from_hosp])
        return sim_mat_new

    def simulate_death(self, sim_mat, sim_mat_new):
        pid_death = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.DEATH)]["pid"]
        if not pid_death.size:
            return sim_mat_new

        sim_mat_new["comp_this" ][pid_death] = self.DEATH
        sim_mat_new["comp_next" ][pid_death] = self.DEATH
        sim_mat_new["cd_trans"  ][pid_death] = np.inf
        sim_mat_new["type_quran"][pid_death] = self.Q_FREE
        return sim_mat_new

    def simulate_susc(self, sim_mat, sim_mat_new):
        pid_susc = sim_mat[(sim_mat["cd_trans"]<0)*(sim_mat["comp_next"]==self.SUSC)]["pid"]

        if not pid_susc.size:
            return sim_mat_new

        sim_mat_new["comp_this"][pid_susc] = self.SUSC
        sim_mat_new["comp_next"][pid_susc] = self.SUSC
        sim_mat_new["cd_trans" ][pid_susc] = np.inf
        return sim_mat_new

    def update_isolation(self, sim_mat):
        pid_iso = sim_mat[sim_mat["cd_quran"]<0]["pid"]
        if not pid_iso.size:
            return sim_mat

        sim_mat["type_quran"][pid_iso] = self.Q_FREE
        sim_mat["cd_quran"  ][pid_iso] = np.inf
        return sim_mat

    def simulate_transition(self, sim_mat):
        pid_trans = sim_mat["pid"][sim_mat["cd_trans"]<0]
        if not pid_trans.size:
            return sim_mat, np.empty(shape=0,dtype=int)

        sim_mat_new = sim_mat.copy()
        while (sim_mat_new["cd_trans"]<0).any():
            sim_mat_new = self.simulate_incb  (sim_mat, sim_mat_new)
            sim_mat_new = self.simulate_symp  (sim_mat, sim_mat_new)
            sim_mat_new = self.simulate_asym  (sim_mat, sim_mat_new)
            sim_mat_new = self.simulate_mild  (sim_mat, sim_mat_new)
            sim_mat_new = self.simulate_seve  (sim_mat, sim_mat_new)
            sim_mat_new = self.simulate_hosp  (sim_mat, sim_mat_new)
            sim_mat_new = self.simulate_icu   (sim_mat, sim_mat_new)
            sim_mat_new = self.simulate_hospr (sim_mat, sim_mat_new)
            sim_mat_new = self.simulate_recv  (sim_mat, sim_mat_new)
            sim_mat_new = self.simulate_death (sim_mat, sim_mat_new)
            sim_mat_new = self.simulate_susc  (sim_mat, sim_mat_new)
            sim_mat = sim_mat_new.copy()
        sim_mat = self.update_isolation(sim_mat)
        assert not np.where(sim_mat["comp_this"]==self.SYMP)[0].size, "Error Occur when Processing Compartment Transition: Compart=SYMP!"
        return sim_mat,pid_trans

    def fromiter_concat(self, it, dtype):
        """
        Clone the iterator to get its length
        https://stackoverflow.com/questions/32997108
        """
        it, it2 = itertools.tee(it)
        flattened = itertools.chain.from_iterable(it)
        array = np.fromiter(flattened, dtype)
        length = np.fromiter(map(len, it2), dtype=int)
        return array, length

    def lookup_contact_home(self, pid_infectious, hdict, place, stype):
        hid_infected = self.tab_person["hid"][pid_infectious]
        it = map(lambda x: hdict.get(x, np.array([], dtype=int)), hid_infected)
        targets, length = self.fromiter_concat(it, int)
        sources = np.repeat(pid_infectious, length)
        places = np.repeat(place, sum(length))
        stypes = np.repeat(stype, sum(length))
        contacts = np.concatenate([sources, targets, places, stypes]).reshape(4,-1).T
        return contacts

    def lookup_contact_occup_(self, pid_infectious, wid_infected, n_c_exp, wdict, place, stype):
        it = map(lambda x: wdict.get(x, np.array([], dtype=int)), wid_infected)
        it, it2 = itertools.tee(it)
        n_c = np.fromiter(map(len, it2), dtype=int)
        np.putmask(n_c, n_c>n_c_exp, n_c_exp)

        targets = map(lambda x: np.random.choice(x[0], x[1]), zip(it, n_c))
        # targets = np.fromiter(itertools.chain.from_iterable(targets), dtype="int32")
        targets_ = list(targets)
        targets = np.concatenate(targets_) if len(targets_) else np.empty(shape=0, dtype=int)
        sources = np.repeat(pid_infectious, n_c)
        places = np.repeat(place, sum(n_c))
        stypes = np.repeat(stype, sum(n_c))
        contacts = np.concatenate([sources, targets, places, stypes]).reshape(4, -1).T
        return contacts

    def lookup_contact_occup(self, pid_infectious, wid_infected, n_c_exp, wdict, place, stype):
        if not pid_infectious.size:
            return np.empty(shape=(0,4), dtype=int)

        it = map(lambda x: wdict.get(x, np.array([], dtype=int)), wid_infected)
        it, it2 = itertools.tee(it)
        w_size = np.fromiter(map(len, it2), dtype=int)
        n_c = w_size.copy()
        np.putmask(n_c, n_c>n_c_exp, n_c_exp)

        index_float = np.random.rand(len(wid_infected), n_c_exp)
        w_size_ = np.repeat(w_size, n_c_exp).reshape(-1, n_c_exp)
        start_idx = np.repeat(np.r_[0, w_size.cumsum()[:-1]], n_c_exp).reshape(-1, n_c_exp)
        index_ = ((w_size_*index_float-1).astype(int)+start_idx).flatten()
        targets = np.concatenate(list(it))[index_]

        sources = np.repeat(pid_infectious, n_c_exp)
        places = np.repeat(place, n_c_exp*len(pid_infectious))
        stypes = np.repeat(stype, n_c_exp*len(pid_infectious))
        contacts = np.concatenate([sources, targets, places, stypes]).reshape(4, -1).T
        return contacts

    # def lookup_contact_other_(self, pid_infectious, hzdict, place, stype):
    #     n_c_exp = self.tab_person["nco_sq"][pid_infectious]
    #     ozones = self.tab_person["h_sq"][pid_infectious]
    #
    #     it = map(lambda x: hzdict.get(x, np.array([], dtype=int)), ozones)
    #     it, it2 = itertools.tee(it)
    #     n_c = np.fromiter(map(len, it2), dtype=int)
    #     np.putmask(n_c, n_c>n_c_exp, n_c_exp)
    #
    #     targets = map(lambda x: np.random.choice(x[0], x[1]), zip(it, n_c))
    #     targets_ = list(targets)
    #     targets = np.concatenate(targets_) if len(targets_) else np.empty(shape=0, dtype=int)
    #     # targets = np.fromiter(itertools.chain.from_iterable(targets), dtype="int32")
    #     sources = np.repeat(pid_infectious, n_c)
    #     places = np.repeat(place, sum(n_c))
    #     stypes = np.repeat(stype, sum(n_c))
    #     contacts = np.concatenate([sources, targets, places, stypes]).reshape(4, -1).T
    #
    #     return contacts

    def lookup_contact_other(self, pid_infectious, hzdict, place, stype):
        if not pid_infectious.size:
            return np.empty(shape=(0,4), dtype=int)

        pid_nco_ozones = pd.DataFrame(self.tab_person[pid_infectious][["pid","h_sq","nco_sq"]], columns=["pid","h_sq","nco_sq"])
        # pid_nco_ozones["nco_sq"] = pid_nco_ozones["nco_sq"].mask(pid_nco_ozones["nco_sq"]>self.npi_max_gather, self.npi_max_gather)               # <----------------
        pid_nco_ozones = pid_nco_ozones.sort_values(by="h_sq")
        pid_infectious = pid_nco_ozones["pid"].values

        dic_ozone_nco = pid_nco_ozones[["h_sq","nco_sq"]].groupby("h_sq")["nco_sq"].apply(sum).to_dict()
        ozone_nco = sorted(dic_ozone_nco.items(), key=lambda item: item[0])

        targets = map(lambda x: np.random.choice(hzdict.get(x[0]),x[1]) if x[0] in hzdict.keys() else np.repeat(-1,x[1]).astype(int), ozone_nco)
        targets = np.concatenate(list(targets))
        sources = np.repeat(pid_infectious, pid_nco_ozones.nco_sq)
        places  = np.repeat(place, len(sources))
        stypes  = np.repeat(stype, len(sources))

        contacts = np.concatenate([sources, targets, places, stypes]).reshape(4,-1).T
        contacts = contacts[contacts[:,1]>0]
        return contacts

    def lookup_contact_other2(self, pid_infectious, hjdict, place, stype):
        if not pid_infectious.size:
            return np.empty(shape=(0,4), dtype=int)

        pid_nco_ozones = pd.DataFrame(self.tab_person[pid_infectious][["pid","h_jd","nco_jd"]], columns=["pid","h_jd","nco_jd"])
        # pid_nco_ozones["nco_sq"] = pid_nco_ozones["nco_sq"].mask(pid_nco_ozones["nco_sq"]>self.npi_max_gather, self.npi_max_gather)               # <----------------
        pid_nco_ozones = pid_nco_ozones.sort_values(by="h_jd")
        pid_infectious = pid_nco_ozones["pid"].values

        dic_ozone_nco = pid_nco_ozones[["h_jd","nco_jd"]].groupby("h_jd")["nco_jd"].apply(sum).to_dict()
        ozone_nco = sorted(dic_ozone_nco.items(), key=lambda item: item[0])

        targets = map(lambda x: np.random.choice(hjdict.get(x[0]),x[1]) if x[0] in hjdict.keys() else np.repeat(-1,x[1]).astype(int), ozone_nco)
        targets = np.concatenate(list(targets))
        sources = np.repeat(pid_infectious, pid_nco_ozones.nco_jd)
        places  = np.repeat(place, len(sources))
        stypes  = np.repeat(stype, len(sources))

        contacts = np.concatenate([sources, targets, places, stypes]).reshape(4,-1).T
        contacts = contacts[contacts[:,1]>0]
        return contacts

    def select_infection_from_contact(self, contacts, prob_trans):
        contacts = contacts[contacts[:,0] != contacts[:,1]]
        if not contacts.size:
            return np.empty(shape=(0,4), dtype=int)

        idx_infect = self._random_binomial_choice(range(contacts.shape[0]), prob_trans)
        infections = contacts[idx_infect]
        return infections

    def get_infection_home(self, pid_infectious, source_type, iso):
        if not pid_infectious.size:
            return np.empty(shape=(0,4), dtype=int)
        prob_trans = self.ptrans*self.ic_setting[self.HOME]*self.infectivity[source_type]
        prob_trans = prob_trans*self.theta_iso if iso else prob_trans
        contacts = self.lookup_contact_home(pid_infectious, self.hdict, self.HOME, source_type)
        infections = self.select_infection_from_contact(contacts, prob_trans)
        return infections

    def get_infection_occup(self, pid_infectious, source_type):
        if not pid_infectious.size:
            return np.empty(shape=(0,4), dtype=int)

        infections = []
        work_types = [self.WORK] if self.is_school_off else [self.WORK,self.KINDER,self.PRIMARY,self.MIDDLE,self.HIGH]


        for wtype in work_types:
            prob_trans = self.ptrans*self.ic_setting[wtype]*self.infectivity[source_type]
            prob_trans = prob_trans*self.r_ic_work if wtype==self.WORK else prob_trans*self.r_ic_school                               # <-----------
            pid_infectious_wtype = pid_infectious[self.tab_person["wtype"][pid_infectious]==wtype]
            n_c_wg = self.n_contact_wgroup[wtype]
            n_c_wp = self.n_contact_wplace[wtype]
            wgid_infected = self.tab_person["wgroup"][pid_infectious_wtype]
            wpid_infected = self.tab_person["wplace"][pid_infectious_wtype]

            contacts_wp = self.lookup_contact_occup(pid_infectious_wtype, wpid_infected, n_c_wp, self.wpdict, wtype, source_type)
            contacts_wg = self.lookup_contact_occup(pid_infectious_wtype, wgid_infected, n_c_wg, self.wgdict, wtype, source_type)
            contacts = np.r_[contacts_wp, contacts_wg]

            infections_wtype = self.select_infection_from_contact(contacts, prob_trans)
            infections.append(infections_wtype)
        infections = np.concatenate(infections)
        return infections

    def get_infection_occup_action(self, pid_infectious, source_type,wzone,hzone):
        if not pid_infectious.size:
            return np.empty(shape = (0, 4), dtype = int)

        infections = []
        work_types = [self.WORK] if self.is_school_off else [self.WORK, self.KINDER, self.PRIMARY, self.MIDDLE,
                                                             self.HIGH]
        tab_person=self.tab_person.copy()
        tab_person['wtype'][np.in1d(tab_person["wplace"], wzone)] = -1

        tab_person['wtype'][np.in1d(tab_person["h_sq"], hzone)] = -1
        #group by by wplace
        for wtype in work_types:
            prob_trans = self.ptrans * self.ic_setting[wtype] * self.infectivity[source_type]
            prob_trans = prob_trans * self.r_ic_work if wtype == self.WORK else prob_trans * self.r_ic_school  # <-----------
            pid_infectious_wtype = pid_infectious[tab_person["wtype"][pid_infectious] == wtype]
            n_c_wg = self.n_contact_wgroup[wtype]
            n_c_wp = self.n_contact_wplace[wtype]
            wgid_infected = tab_person["wgroup"][pid_infectious_wtype]
            wpid_infected = tab_person["wplace"][pid_infectious_wtype]

            contacts_wp = self.lookup_contact_occup(pid_infectious_wtype, wpid_infected, n_c_wp, self.wpdict, wtype,
                                                    source_type)
            contacts_wg = self.lookup_contact_occup(pid_infectious_wtype, wgid_infected, n_c_wg, self.wgdict, wtype,
                                                    source_type)
            contacts = np.r_[contacts_wp, contacts_wg]

            infections_wtype = self.select_infection_from_contact(contacts, prob_trans)
            infections.append(infections_wtype)
        infections = np.concatenate(infections)
        # print(infections)
        # print(111111111111)
        return infections

    def get_infection_other(self, pid_infectious, source_type):
        if not pid_infectious.size:
            return np.empty(shape=(0,4), dtype=int)

        prob_trans = self.ptrans*self.ic_setting[self.OTHER]*self.infectivity[source_type]*self.r_ic_other                   # <--------------
        contacts = self.lookup_contact_other(pid_infectious, self.hzdict, self.OTHER, source_type)
        infections = self.select_infection_from_contact(contacts, prob_trans)
        return infections

    def get_infection_other2(self, pid_infectious, source_type):
        if not pid_infectious.size:
            return np.empty(shape=(0,4), dtype=int)

        prob_trans = self.ptrans*self.ic_setting[self.OTHER]*self.infectivity[source_type]*self.r_ic_other                   # <--------------
        contacts = self.lookup_contact_other2(pid_infectious, self.hjdict, self.OTHER, source_type)
        infections = self.select_infection_from_contact(contacts, prob_trans)
        return infections

    def get_infection_by_source_type(self, sim_mat, pid_infectious, source_type):
        """
        将发病前/无症状/有症状的具有感染力个体按照不同 Quarantine 的状态来分组
        并根据 Quarantine 状态来决定是否参与 Household/Workplace/Community网络的传播
        """
        if not pid_infectious.size:
            return np.empty(shape=(0,4), dtype=int)

        pid_infectious_q_free  = pid_infectious[sim_mat["type_quran"][pid_infectious]==self.Q_FREE ]
        # pid_infectious_q_iso_h = pid_infectious[sim_mat["type_quran"][pid_infectious]==self.Q_ISO_H]
        pid_infectious_Q_ISO_HOME = pid_infectious[sim_mat["type_quran"][pid_infectious]==self.Q_ISO_HOME]
        pid_infectious_q_iso   =  pid_infectious_Q_ISO_HOME
        # pid_infectious_q_iso   = np.r_[pid_infectious_q_iso_h, pid_infectious_Q_ISO_HOME]
        pid_infectious_q_com=pid_infectious[sim_mat["type_quran"][pid_infectious]==self.Q_COM]
        pid_infectious_com=np.r_[pid_infectious_q_free,pid_infectious_q_com]
        infections_home  = self.get_infection_home (pid_infectious_com, source_type=source_type, iso=False)
        infections_home = infections_home[sim_mat['type_quran'][infections_home[:,1]]<=self.Q_ISO_HOME]
        infections_iso   = self.get_infection_home (pid_infectious_q_iso,  source_type=source_type, iso=True )
        infections_iso = infections_iso[sim_mat['type_quran'][infections_iso[:,1]]<=self.Q_ISO_HOME]
        infections_occup = self.get_infection_occup(pid_infectious_q_free, source_type=source_type)
        infections_occup = infections_occup[sim_mat['type_quran'][infections_occup[:,1]]==self.Q_FREE]
        infections_other = self.get_infection_other(pid_infectious_com, source_type=source_type)
        infections_other = infections_other[(sim_mat['type_quran'][infections_other[:,1]]==self.Q_FREE)|(sim_mat['type_quran'][infections_other[:,1]]==self.Q_COM)]

        infections_other2 = self.get_infection_other2(pid_infectious_q_free, source_type = source_type)
        infections_other2 = infections_other2[(sim_mat['type_quran'][infections_other2[:,1]]==self.Q_FREE)|(sim_mat['type_quran'][infections_other2[:,1]]==self.Q_COM)]

        infections = np.concatenate([infections_home, infections_iso, infections_occup, infections_other,infections_other2])
        return infections


    def simulate_infection_blocking_by_vaccine(self, sim_mat, infections):
        if not infections.size:
            return infections, np.empty(shape=0, dtype=int)

        sim_dates, sources, targets, places, stypes, days_infect = [*infections.T]
        idx_latt, idx_blocked = self.get_pid_with_outcome(sim_mat, targets, "susc2latt", index_only=True)
        infections = infections[idx_latt]
        pid_blocked = targets[idx_blocked]
        return infections, pid_blocked

    def get_infection(self, sim_mat,sim_date):
        pid_infectious_incb = sim_mat[sim_mat["comp_this"]==self.INCB]["pid"]
        pid_infectious_asym = sim_mat[sim_mat["comp_this"]==self.ASYM]["pid"]
        pid_infectious_mild = sim_mat[sim_mat["comp_this"]==self.MILD]["pid"]

        infections_incb = self.get_infection_by_source_type(sim_mat, pid_infectious_incb, source_type=self.ST_PRES)
        infections_asym = self.get_infection_by_source_type(sim_mat, pid_infectious_asym, source_type=self.ST_ASYM)
        infections_mild = self.get_infection_by_source_type(sim_mat, pid_infectious_mild, source_type=self.ST_SYMP)
        infections = np.concatenate([infections_incb, infections_asym, infections_mild])

        if not infections.size:
            return np.empty(shape=(0,6), dtype=int), np.empty(shape=0, dtype=int),np.empty(shape=(0,4), dtype=int)

        infections  = infections[sim_mat["comp_this"][infections[:,1]]==self.SUSC]                                # 选取 target 为易感者的 infection
        infections_case=infections #所有感染可能
        infections  = np.random.permutation(infections)
        infections  = infections[np.unique(infections[:,1], return_index=True)[1]]                                # 个体单日被多次感染，随机选取一个
        sim_dates   = np.repeat(sim_date, infections.shape[0])
        days_infect = sim_mat["days_infect"][infections[:,0]]                                                     # 父代感染天数 (计算代际间隔、Rt等)
        infections  = np.c_[sim_dates, infections, days_infect]

        if self.is_ve_anti_infection:
            infections, pid_blocked = self.simulate_infection_blocking_by_vaccine(sim_mat, infections)
        else:
            pid_blocked = np.empty(shape=0, dtype=int)
        return infections, pid_blocked,infections_case

    def simulate_infection(self, sim_mat, infections):
        if not infections.size:
            return sim_mat

        sim_dates, sources, targets, places, stypes, days_infect = [*infections.T]
        sim_mat["comp_this"  ][targets] = self.LATT
        sim_mat["comp_next"  ][targets] = self.INCB
        sim_mat["cd_trans"   ][targets] = self.make_duration_lognorm("latt2incb", len(targets))
        sim_mat["days_infect"][targets] = 0
        return sim_mat

    def simulate_close_class(self, sim_mat):
        if self.is_close_class:
            sim_mat_ = sim_mat[self.pid_students]
            pid_mild = sim_mat_[sim_mat_["comp_this"]==self.MILD]["pid"]
            pid_seve = sim_mat_[sim_mat_["comp_this"]==self.SEVE]["pid"]
            pid_report = np.r_[self._random_binomial_choice(pid_mild, self.rate_onset_report), pid_seve]

            counter_wg = Counter(self.tab_person["wgroup"][pid_report])
            wgroups = np.array([k for k, v in counter_wg.items() if v>self.th_close_class])
            pid_close_class = np.where(np.isin(self.tab_person["wgroup"], wgroups))[0]
            sim_mat["type_quran"][pid_close_class] = self.Q_CLASS
            sim_mat["cd_quran"  ][pid_close_class] = self.days_close_class
        else:
            wgroups = np.empty(shape=0, dtype=int)
        return sim_mat, wgroups

    def get_date_onset(self, pid, date_infected, pid_onset_concat, n_daily_onset_cusm, s_days, e_days):
        s_idx = n_daily_onset_cusm[date_infected]+s_days
        e_idx = n_daily_onset_cusm[min(date_infected+e_days, len(n_daily_onset_cusm)-1)]
        pids = pid_onset_concat[s_idx:e_idx]

        idx = np.where(pids==pid)[0]

        if idx.size:
            date_onset = np.searchsorted(n_daily_onset_cusm, idx+s_idx, side="right")[0]
        else:
            date_onset = -9999
        return date_onset

    def process_sim_result(self, sim_res):
        keys = ["daily_onset", "daily_hosp", "daily_icu", "daily_recv", "daily_death",
                "daily_infect", "daily_blocked", "current_onset", "current_hosp",
                "current_icu", "current_recv", "current_death", "current_infect", "current_quran"]
        for k in keys:
            sim_res[k] = np.array(sim_res[k])

        for k in ["daily_counter", "current_counter"]:
            sim_res[k] = np.concatenate(sim_res[k])

        date_inf_t, sources, targets, places, stypes, GT = [*sim_res["infections"].T]
        date_inf_s = date_inf_t-GT

        if self.is_fast_r0:
            date_onset_s = date_inf_s
            date_onset_t = date_inf_t
        else:
            pid_onset_concat = np.concatenate(sim_res["pid_onset"])
            n_daily_onset_cusm = np.cumsum(np.fromiter(map(len,sim_res["pid_onset"]), dtype=int))

            test_duration = self.make_duration_lognorm("latt2incb",self.n_agent)+self.make_duration_lognorm("incb2symp",self.n_agent)
            s_days, e_days = int(min(test_duration)), int(max(test_duration))+1

            date_onset_s = np.fromiter(map(lambda x: self.get_date_onset(x[0],x[1],pid_onset_concat,n_daily_onset_cusm,s_days,e_days),zip(sources,date_inf_s)), dtype=int)
            date_onset_t = np.fromiter(map(lambda x: self.get_date_onset(x[0],x[1],pid_onset_concat,n_daily_onset_cusm,s_days,e_days),zip(targets,date_inf_t)), dtype=int)

        sim_res["metadata"]["date_pid_imported"] = np.concatenate(sim_res["metadata"]["date_pid_imported"])
        sim_res["infections"] = np.c_[sim_res["infections"], date_inf_s, date_onset_s, date_onset_t]
        inf_dtype = np.dtype([("date_inf_t", "int16"), ("sources", "int32"), ("targets", "int32"), ("places", "int8"),
                              ("stypes", "int8"), ("GT", "int16"), ("date_inf_s", "int16"), ("date_onset_s", "int16"), ("date_onset_t", "int32")])
        sim_res["infections"] = rfn.unstructured_to_structured(sim_res["infections"], inf_dtype)
        return sim_res

    def init_sim_res(self, sim_mat):
        """
        初始化并记录 Day 0 的模型状态
        """
        sim_res = defaultdict(list)
        sim_res["metadata"] = self.__get_metadata__()
        sim_res["infections"] = np.empty(shape=(0,6), dtype=int)                                          # infections 在追溯密接时用到, 需要先进行结构化, 不能在最后处理sim_res时concatenate
        # sim_res["patient_outside"] = defaultdict(list)                                                    # 由于春运未能回到深圳的患者 (含死亡)
        return sim_res

    def update_sim_result_statistic(self, sim_res):
        # record age-specified death/hospitalization/ICU rate
        age_demo   = np.bincount(self.tab_person["age"])                                                  # 统计模型中所有agent年龄占比
        age_infect = np.bincount(self.tab_person["age"][np.concatenate(sim_res["pid_infect"])])           # 统计模型中所有被感染个体的年龄占比
        age_onset  = np.bincount(self.tab_person["age"][np.concatenate(sim_res["pid_onset" ])])           # 统计模型中所有发病个体的年龄占比
        age_hosp   = np.bincount(self.tab_person["age"][np.concatenate(sim_res["pid_hosp"  ])])           # 统计模型中所有住院个体的年龄占比
        age_icu    = np.bincount(self.tab_person["age"][np.concatenate(sim_res["pid_icu"   ])])           # 统计模型中所有ICU个体的年龄占比
        age_death  = np.bincount(self.tab_person["age"][np.concatenate(sim_res["pid_death" ])])           # 统计模型中所有死亡个体的年龄占比
        age_quran  = np.bincount(self.tab_person["age"][np.concatenate(sim_res["pid_quran" ])])           # 统计模型中所有死亡个体的年龄占比

        for stat in [age_demo, age_infect, age_onset, age_hosp, age_icu, age_death, age_quran]:
            stat.resize(self.n_age_group, refcheck=False)

        sim_res["age_demo"  ] = age_demo
        sim_res["age_infect"] = age_infect
        sim_res["age_onset" ] = age_onset
        sim_res["age_hosp"  ] = age_hosp
        sim_res["age_icu"   ] = age_icu
        sim_res["age_death" ] = age_death

        # calculate Rt/R0/SI/GT/Household SAR
        infection_onset = sim_res["infections"][(sim_res["infections"]["date_onset_s"]>0)*(sim_res["infections"]["date_onset_t"]>0)]
        if infection_onset.size:
            SI_array = infection_onset["date_onset_t"]-infection_onset["date_onset_s"]
            SI = np.mean(SI_array)
            GT = np.mean(sim_res["infections"]["GT"])
        else:
            SI = GT = np.NaN

        n_sources = np.bincount(sim_res["infections"]["date_inf_t"])
        n_targets = np.bincount(sim_res["infections"]["date_inf_s"])
        n_imported_day0 = np.array([self.n_imported_day0])
        n_sources.resize(self.max_iterday+1, refcheck=False)
        n_targets.resize(self.max_iterday+1, refcheck=False)

        self.date_pid_imported = np.concatenate(self.date_pid_imported)
        sim_res["imported_seed"] = self.date_pid_imported
        n_imported = np.bincount(self.date_pid_imported[:,0])
        n_imported.resize(self.max_iterday+1, refcheck=False)
        n_sources = n_sources+n_imported

        np.seterr(invalid='ignore')
        Rt = n_targets/n_sources
        sim_res["Rt"] = Rt
        sim_res["R0"] = Rt[0]
        sim_res["SI"] = SI
        sim_res["GT"] = GT
        return sim_res

    def update_sim_result_daily(self, sim_res, sim_mat, pid_trans, daily_infections, pid_blocked, wg_close):
        sim_dates,sources,targets,places,stypes,days_infect = [*daily_infections.T]

        daily_counter = np.bincount(sim_mat["comp_this"][pid_trans])
        daily_counter.resize(len(self.COMP_NAMES), refcheck=False)
        daily_counter[self.LATT] = daily_infections.shape[0]+self.n_imported_daily           # 其它仓室在转移时间为0时变化,但LATT由infection触发,需要单独记录
        sim_res["daily_counter"].append([daily_counter])
        curr_counter = np.bincount(sim_mat["comp_this"])                           # 各仓室的每日现存人数记录
        curr_counter.resize(len(self.COMP_NAMES), refcheck=False)
        sim_res["current_counter"].append([curr_counter])
        sim_mat_new=sim_mat.copy()

        # sim_mat_new["type_quran"][sim_mat_new["type_quran"]==self.Q_HOSP]=2
        quran_counter = np.bincount(sim_mat_new["type_quran"])                         # 各种quarantine状态的每日现存人数记录
        quran_counter.resize(3, refcheck=False)

        sim_res["quran_counter"].append([quran_counter])
        # record person ID within different compartment
        sim_mat_  = sim_mat[pid_trans]
        pid_latt  = daily_infections[:, 2]
        pid_onset = sim_mat_["pid"][(sim_mat_["comp_this"]==self.MILD)+(sim_mat_["comp_this"]==self.SEVE )]
        pid_hosp  = sim_mat_["pid"][(sim_mat_["comp_this"]==self.HOSP)+(sim_mat_["comp_this"]==self.HOSPR)]
        pid_icu   = sim_mat_["pid"][(sim_mat_["comp_this"]==self.ICU  )]
        pid_death = sim_mat_["pid"][(sim_mat_["comp_this"]==self.DEATH)]
        # print("hosptial ID:",sim_mat['pid'][sim_mat["type_quran"]==self.Q_HOSP])
        sim_res["pid_infect"].append(pid_latt )
        sim_res["pid_onset" ].append(pid_onset)
        sim_res["pid_hosp"  ].append(pid_hosp )
        sim_res["pid_icu"   ].append(pid_icu  )
        sim_res["pid_death" ].append(pid_death)

        # record daily onset cases in different communities
        # counter_onset_jd = np.bincount(self.tab_person["h_jd"][pid_onset])
        # sim_res["daily_onset_jd"].append(counter_onset_jd)

        # record daily simulation result
        sim_res["daily_onset"   ].append(daily_counter[self.MILD ]+daily_counter[self.SEVE ])
        sim_res["daily_hosp"    ].append(daily_counter[self.HOSP ]+daily_counter[self.HOSPR])
        sim_res["daily_icu"     ].append(daily_counter[self.ICU  ])
        sim_res["daily_recv"    ].append(daily_counter[self.RECV ])
        sim_res["daily_death"   ].append(daily_counter[self.DEATH])
        sim_res["daily_infect"  ].append(daily_counter[self.LATT ])
        sim_res["daily_blocked" ].append(int(len(pid_blocked)))

        # record current simulation result
        sim_res["current_onset" ].append(curr_counter[self.MILD ]+curr_counter[self.SEVE ])
        sim_res["current_hosp"  ].append(curr_counter[self.HOSP ]+curr_counter[self.HOSPR])
        sim_res["current_icu"   ].append(curr_counter[self.ICU  ])
        sim_res["current_recv"  ].append(curr_counter[self.RECV ])
        sim_res["current_death" ].append(curr_counter[self.DEATH])
        sim_res["current_infect"].append(sim_mat.shape[0]-(curr_counter[self.SUSC]+curr_counter[self.DEATH]))

        # record daily quarantined agent
        # pid_quran = sim_mat["pid"][(sim_mat["type_quran"]==self.Q_ISO_H)+(sim_mat["type_quran"]==self.Q_ISO_HOME)]
        pid_quran = sim_mat["pid"][(sim_mat["type_quran"]==self.Q_ISO_HOME)]
        sim_res["pid_quran"     ].append(pid_quran)
        sim_res["current_quran" ].append(len(pid_quran))
        sim_res["wg_close"      ].append(wg_close)

        # record daily infection event
        sim_res["infections"] = np.concatenate([sim_res["infections"], daily_infections], axis=0)

        # record final status
        # sim_res["sim_mat"] = sim_mat                                              # 动态表的最终状态 存储占用过大 一般不存
        sim_res["levels"].append(self.temp_level)
        return sim_res

    def simulate_dynamic_npi_61(self,sim_mat,sim_res):

        """
        根据ICU占用率 触发不同npi

        跨级调整版本（level1-level3-level1）

        触发阈值:th_icu_on-th_icu_off

        阶段phase0-1-2-3-4分别代表0-1之间，1-2之间……

        """
        # 单一level PHSM 或已降至low_level 或模拟天数未达到7天
        if self.temp_phase == 2:
            return sim_res

        # 采样网络的icu占用数及占用率
        n_icu = np.sum([sim_mat["comp_this"] == self.ICU])
        true_rate_occupied_icu = n_icu * self.scale_factor / self.n_cap_icu

        # 平滑后的icu占用数0-sim_date-3，及sim_date-3的占用率
        smoothed_icu = np.convolve(np.r_[sim_res["current_icu"], [n_icu]], np.ones(7) / 7, mode='valid')
        rate_occupied_icu = smoothed_icu[-1] * self.scale_factor / self.n_cap_icu
        if self.temp_phase == 0 and true_rate_occupied_icu > self.th1:
            self.temp_phase = 1
            self.temp_level = 3
            self.set_PHSM_params()
            print(self.sim_date, "icu up to", true_rate_occupied_icu, "level 3 on")
            sim_res['info'].append(
                "sim_date: %s,icu_rate: %s,level: %s" % (self.sim_date, true_rate_occupied_icu, self.temp_level))
            return sim_res
        if self.temp_phase == 1 and not self.off_allowance and true_rate_occupied_icu >= self.th2:
            self.off_allowance = True
            return sim_res
        if self.temp_phase == 1 and self.off_allowance and true_rate_occupied_icu < self.th2:
            self.temp_phase = 2
            self.temp_level = 1
            self.set_PHSM_params()
            print(self.sim_date, "icu up to", true_rate_occupied_icu, "level 1 on")
            sim_res['info'].append(
                "sim_date: %s,icu_rate: %s,level: %s" % (self.sim_date, true_rate_occupied_icu, self.temp_level))
            return sim_res

        return sim_res

    def simulate_dynamic_npi(self, sim_mat, sim_res):

        """
        根据ICU占用率 触发不同npi

        逐级调整版本（level1-level3-level4-level2-level1）

        触发阈值th1-th2-th3-th4

        阶段phase0-1-2-3-4分别代表0-1之间，1-2之间……

        """
        # 单一level PHSM 或已降至low_level 或模拟天数未达到7天
        if self.temp_phase == 4:
            return sim_res

        # 采样网络的icu占用数及占用率
        n_icu = np.sum([sim_mat["comp_this"] == self.ICU])
        true_rate_occupied_icu = n_icu * self.scale_factor / self.n_cap_icu

        # 平滑后的icu占用数0-sim_date-3，及sim_date-3的占用率
        smoothed_icu = np.convolve(np.r_[sim_res["current_icu"], [n_icu]], np.ones(7) / 7, mode='valid')
        rate_occupied_icu = smoothed_icu[-1] * self.scale_factor / self.n_cap_icu

        if self.temp_phase == 0 and true_rate_occupied_icu > self.th1:
            self.temp_phase = 1
            self.temp_level = 3
            self.set_PHSM_params()
            print(self.sim_date, "icu up to", true_rate_occupied_icu, "level 3 on")
            sim_res['info'].append(
                "sim_date: %s,icu_rate: %s,level: %s" % (self.sim_date, true_rate_occupied_icu, self.temp_level))
            return sim_res
        if self.temp_phase == 1 and true_rate_occupied_icu > self.th2:
            self.temp_phase = 2
            self.temp_level = 4
            self.set_PHSM_params()
            print(self.sim_date, "icu up to", true_rate_occupied_icu, "level 4 on")
            sim_res['info'].append(
                "sim_date: %s,icu_rate: %s,level: %s" % (self.sim_date, true_rate_occupied_icu, self.temp_level))
            return sim_res
        if self.temp_phase == 2:
            self.level4days += 1

            if true_rate_occupied_icu > self.th3 and not self.off_allowance:
                self.off_allowance = True
            if (self.off_allowance and true_rate_occupied_icu < self.th3) or self.level4days >= 20:
                self.off_allowance = False
                self.temp_phase = 3
                self.temp_level = 2
                self.set_PHSM_params()
                print(self.sim_date, "icu down to", true_rate_occupied_icu, "level 2 on")
                sim_res['info'].append(
                    "sim_date: %s,icu_rate: %s,level: %s" % (self.sim_date, true_rate_occupied_icu, self.temp_level))
                return sim_res
        if self.temp_phase == 3 and true_rate_occupied_icu < self.th4:
            self.temp_phase = 4
            self.temp_level = 1
            self.set_PHSM_params()
            print(self.sim_date, "icu down to", true_rate_occupied_icu, "level 1 on")
            sim_res['info'].append(
                "sim_date: %s,icu_rate: %s,level: %s" % (self.sim_date, true_rate_occupied_icu, self.temp_level))
            return sim_res


        # 平滑后的占用率已下降，且真实最高占用率小于解除阈值，即从未达到解除阈值且已过峰的情况，仅用于深圳！
        # if self.city_name == "Shenzhen" and self.temp_level == self.high_level and smoothed_icu[-1] < max(
        #         smoothed_icu) * 0.9 and max(
        #         sim_res["current_icu"]) * self.scale_factor / self.n_cap_icu < self.th_icu_off:
        #     self.icu_peak_arrived = True
        #     print(self.sim_date, "icu down from peak,smoothed:", rate_occupied_icu, true_rate_occupied_icu)
        #     print("感染规模:", sum(sim_res["daily_infect"]) / self.n_agent)
        #
        #     self.temp_level = self.low_level
        #     self.set_PHSM_params()
        #     self.no_more_change = True
        #     sim_res['info'].append("%s icu down from peak,icu_rate: %s" % (self.sim_date, true_rate_occupied_icu))
        #     return sim_res

        # icu占用率降至解除阈值以下，npi降为low level
        # if not self.is_pop_off and self.temp_level == self.high_level and self.off_allowance \
        #         and true_rate_occupied_icu < self.th_icu_off and smoothed_icu[-1] < max(smoothed_icu):
        return sim_res

    def get_temporal_iterator(self):
        """跨平台: 本机上使用tqdm进度条, 集群上不显示进度"""
        # try:
        #     from tqdm import tqdm
        #     temporal_iterator = tqdm(range(1, self.max_iterday+1))
        # except:
        #     temporal_iterator = range(1, self.max_iterday+1)

        temporal_iterator = range(1, self.max_iterday+1)
        return temporal_iterator

    def get_r_comp_by_age(self, sim_mat, age):
        """ 获取当前状态下模拟城市分年龄段的仓室构成比例 """
        compartments = sim_mat["comp_this"][self.tab_person["age"]==age]
        compartments = compartments[compartments!=self.OUT]
        r_comp = np.bincount(compartments)
        r_comp.resize(len(self.COMP_NAMES), refcheck=False)
        r_comp = r_comp/sum(r_comp)
        return r_comp

    # @profile
    def simulate_comp_move_in(self, r_comp, pid_move_in):
        """ 根据当前城市分年龄段的仓室分布比例模拟春运后返回的个体状态 """
        # r_comp = np.repeat(1, len(self.COMP_NAMES))/len(self.COMP_NAMES)
        r = np.random.uniform(0, 1, pid_move_in.shape[0])
        s = np.searchsorted(r_comp.cumsum(), r)
        return s

    # @profile
    def simulate_migration(self, sim_mat, sim_res):
        """模拟春运活动中的迁入迁出"""
        comp_patient = [self.DEATH, self.ICU, self.HOSP, self.HOSPR]                            # 春运后无法返深的个体仓室类型
        pid_move_out = self.tab_person["pid"][self.tab_person["day_l"]==self.sim_date]
        sim_mat["comp_this"][pid_move_out] = self.OUT
        sim_mat["comp_next"][pid_move_out] = self.OUT
        sim_mat["cd_trans" ][pid_move_out] = np.inf

        pid_move_in = self.tab_person["pid"][self.tab_person["day_r"]==self.sim_date]
        assert (sim_mat["comp_this"][pid_move_in]==self.OUT).all

        if pid_move_in.size:
            sim_mat_move_in = sim_mat[pid_move_in]
            tab_person_move_in = self.tab_person[pid_move_in]

            counter_patient = []
            for age in np.unique(tab_person_move_in["age"]):
                pid_move_in_this_age = tab_person_move_in["pid"][tab_person_move_in["age"]==age]
                r_comp = self.get_r_comp_by_age(sim_mat, age)
                comps = self.simulate_comp_move_in(r_comp, pid_move_in_this_age)

                flg = np.isin(comps, comp_patient)
                pid_patient = pid_move_in[np.where(flg)]
                pid_return = pid_move_in[np.where(~flg)]

                counter_patient_age = np.bincount(comps[np.where(flg)])
                counter_patient_age.resize(len(self.COMP_NAMES))
                counter_patient.append(counter_patient_age)

                sim_mat["comp_next"][pid_return] = comps[np.where(~flg)]
                sim_mat["cd_trans" ][pid_return] = 1                                         # 返深后赋予新的仓室,次日生效

            sim_res["counter_outside"].append(np.sum(counter_patient, axis=0))               # 记录因病/故未能返深的个体仓室统计
        return sim_mat,sim_res

    def reset(self,seed):
        np.random.seed(seed)
        self.sim_date=0
        sim_mat = self.init_sim_mat()
        sim_mat = self.simulate_vaccination(sim_mat)
        sim_mat = self.simulate_elderly_protection(sim_mat)
        sim_mat = self.simulate_drug_access(sim_mat)
        sim_res = self.init_sim_res(sim_mat)
        sim_mat = self.simulate_imported(sim_mat, 0)
        # sim_mat, pid_trans = self.simulate_transition(sim_mat)
        self.get_stat_features()
        individual_state,region_state,prob,region_controlled = self.get_situation_features(sim_mat,self.sim_date)
        self.region_num=self.tab_person_rank.groupby('ranked_hzone').size().to_numpy()

        self.average_infected = np.zeros(self.max_iterday+1)
        region_infected = np.zeros(region_controlled.shape[0])
        state=(individual_state,region_state)
        return state,prob,sim_mat,sim_res,region_infected,region_controlled

    def step(self,sim_mat,sim_res,action):
        self.sim_date += 1
        sim_mat = self.update_days(sim_mat)
        sim_mat = self.simulate_individual_action(sim_mat,action)
        pid_confine=np.where(sim_mat['type_quran']==self.Q_COM)[0]
        self.tab_person['nco_sq']=self.nco_sq_init
        self.tab_person['nco_sq'][pid_confine]=self.nco_sq_init[pid_confine]+self.nco_jd_init[pid_confine]
        # sim_mat, pid_trans = self.simulate_transition(sim_mat)
        # iso_wzone=np.where(wzone_action==self.Q_WORK_ISO)
        # iso_hzone=np.where(hzone_action==self.Q_COM_ISO)
        # index_wzone=self.wplace_index[iso_wzone]
        # index_hzone=self.hzone_index[iso_hzone]
        # daily_infections, pid_blocked = self.get_infection_action(sim_mat,sim_date,index_wzone, index_hzone)
        daily_infections, pid_blocked,infection_case=self.get_infection(sim_mat,self.sim_date)

        # pid_infect_num = np.zeros(len(sim_mat))
        # unique_values, counts = np.unique(infection_case[:, 0], return_counts = True)
        # pid_infect_num[unique_values] = counts
        # pid_infect_home_num = np.zeros(len(sim_mat))
        # unique_values_home, counts_home = np.unique(infection_case[infection_case[:, 2] == -1, 0], return_counts = True)
        # unique_values_home_iso=sim_mat['type_quran'][unique_values_home]==self.Q_ISO_HOME
        # pid_infect_home_num[unique_values_home] = counts_home
        # if len(counts_home[unique_values_home_iso]) > 0:
        #     mean_pid_infect_home = np.sum(counts_home[unique_values_home_iso]) / np.sum(
        #         (sim_mat['comp_this'] > 1) & (sim_mat['comp_this'] < 7) & (sim_mat['type_quran'] == self.Q_ISO_HOME))
        # else:
        #     mean_pid_infect_home = 0
        # unique_values_out, counts_out = np.unique(infection_case[infection_case[:, 2] > -1, 0], return_counts = True)
        # if len(counts_out) > 0:
        #     mean_pid_infect_out = np.sum(counts_out) / np.sum(
        #         (sim_mat['comp_this'] > 1) & (sim_mat['comp_this'] < 7) & (sim_mat['type_quran'] ==self.Q_FREE))
        # else:
        #     mean_pid_infect_out = 0
        sim_mat = self.simulate_infection(sim_mat, daily_infections)
        sim_mat, pid_trans = self.simulate_transition(sim_mat)
        sim_mat, wg_close = self.simulate_close_class(sim_mat)
        sim_res = self.simulate_dynamic_npi_61(sim_mat, sim_res)
        sim_mat = self.simulate_imported(sim_mat, self.sim_date)
        sim_res = self.update_sim_result_daily(sim_res, sim_mat, pid_trans, daily_infections, pid_blocked, wg_close)

        individual_state,region_state,prob,region_controlled=self.get_situation_features(sim_mat,self.sim_date)
        state=(individual_state,region_state)
        unique, counts=np.unique(daily_infections[:,1], return_counts = True)
        pid_infect_count=np.zeros(len(sim_mat))
        pid_infect_count[unique]=counts
        self.tab_person_rank['pid_infect_count']=pid_infect_count
        region_infected=self.tab_person_rank.groupby('ranked_hzone')['pid_infect_count'].sum().to_numpy()
        # self.region_infected[self.sim_date,:]=region_infected

        # infected_num_t=np.sum((sim_mat['comp_this']>0)&(sim_mat['comp_this']<10))+1e-5
        #
        # self.average_infected[self.sim_date]=np.sum(region_infected)/infected_num_t

        # prob[np.where(sim_mat['comp_this']>=4)[0]]=0
        # print(self.average_infected[self.sim_date])
        # if self.sim_date>99:
        #     self.calulate_Rt(sim_res)
        # unique_values, counts = np.unique(infection_case[infection_case[:, 2]>-1,0],return_counts = True)
        # if len(counts)>0:
        #     mean_pid_infect_work= np.mean(counts)
        # else:in
        #     mean_pid_infect_work=0
        return sim_mat, sim_res, state,prob,region_infected,region_controlled


    def calulate_Rt(self,sim_res):
        sim_res1=sim_res.copy()

        date_inf_t, sources, targets, places, stypes, GT = [*sim_res1["infections"].T]
        date_inf_s = date_inf_t - GT

        sim_res1["infections"] = np.c_[sim_res1["infections"], date_inf_s]
        inf_dtype = np.dtype(
            [("date_inf_t", "int16"), ("sources", "int32"), ("targets", "int32"), ("places", "int8"),
             ("stypes", "int8"), ("GT", "int16"), ("date_inf_s", "int16")])
        sim_res1["infections"] = rfn.unstructured_to_structured(sim_res1["infections"], inf_dtype)


        n_sources = np.bincount(sim_res1["infections"]["date_inf_t"])
        n_targets = np.bincount(sim_res1["infections"]["date_inf_s"])
        n_sources.resize(self.max_iterday + 1, refcheck = False)
        n_targets.resize(self.max_iterday + 1, refcheck = False)

        date_pid_imported = np.concatenate(self.date_pid_imported)
        sim_res1["imported_seed"] = date_pid_imported
        n_imported = np.bincount(date_pid_imported[:, 0])
        n_imported.resize(self.max_iterday + 1, refcheck = False)
        n_sources = n_sources + n_imported
        n_sources[n_sources==0]=1
        np.seterr(invalid = 'ignore')
        Rt = n_targets / n_sources
        return Rt


    def simulate_individual_action(self, sim_mat,action):
        """模拟行动措施"""

        # action =action[self.tab_person_rank['ranked_hzone'].to_numpy()]
        pid_in_hosp=(sim_mat["type_quran"]==self.Q_HOSP)
        # sim_mat["type_quran"][np.where(sim_mat["type_quran"][pid_iso_hosp]!=self.Q_HOSP)[0]] = self.Q_ISO
        # sim_mat["type_quran"][np.where(sim_mat["type_quran"][pid_iso_free]!=self.Q_HOSP)[0]] = self.Q_FREE
        # sim_mat["type_quran"][np.where(sim_mat["type_quran"][pid_iso_p]!=self.Q_HOSP)[0]]= self.Q_ISO_HOME

        sim_mat["type_quran"]=action

        sim_mat["type_quran"][pid_in_hosp] = self.Q_HOSP
        sim_mat["type_quran"][(sim_mat["comp_this"]>3)&(sim_mat["comp_this"]<7)] = self.Q_ISO
        sim_mat["type_quran"][(sim_mat["comp_this"]>=10)] = self.Q_FREE


        return sim_mat
    def run(self):
        sim_mat = self.init_sim_mat()
        self.sim_date=0
        sim_mat = self.simulate_vaccination(sim_mat)
        # self.get_stat_features(sim_mat)
        # self.get_situation_features(sim_mat,self.sim_date)
        sim_mat = self.simulate_elderly_protection(sim_mat)
        sim_mat = self.simulate_drug_access(sim_mat)
        sim_res = self.init_sim_res(sim_mat)

        sim_mat = self.simulate_imported(sim_mat, 0)
        for self.sim_date in self.get_temporal_iterator():
            print ("Day ",self.sim_date)
            # if self.sim_date==self.days_school_off:
            #     temp_ic_primary = self.ic_setting[self.PRIMARY]
            #     temp_ic_middle  = self.ic_setting[self.MIDDLE ]
            #     temp_ic_high    = self.ic_setting[self.HIGH   ]
            #     self.ic_setting[self.PRIMARY]=self.ic_setting[self.MIDDLE]=self.ic_setting[self.HIGH]=0
            # if self.sim_date==self.days_school_on:
            #     self.ic_setting[self.PRIMARY] = temp_ic_primary
            #     self.ic_setting[self.MIDDLE ] = temp_ic_middle
            #     self.ic_setting[self.HIGH   ] = temp_ic_high

            # sim_mat,sim_res = self.simulate_migration(sim_mat, sim_res)

            sim_mat = self.update_days(sim_mat)
            # sim_mat,pid_trans = self.simulate_transition(sim_mat)
            # self.tab_person_rank.groupby('ranked_jdzone')['ranked_hzone'].unique()
            # jd_hzone_contact=torch.zeros((63,587),dtype=torch.float16)
            # for i in range(63):
            #     jd_hzone_contact[i,self.tab_person_rank.groupby('ranked_jdzone')['ranked_hzone'].unique()[i]]=1
            # torch.save(jd_hzone_contact,'/home/yxluo/research/EPC_RF/env/jd_hzone_contact.pt')
            daily_infections, pid_blocked,infection_case = self.get_infection(sim_mat,self.sim_date)

            sim_mat = self.simulate_infection(sim_mat, daily_infections)
            sim_mat, pid_trans = self.simulate_transition(sim_mat)
            sim_mat,wg_close = self.simulate_close_class(sim_mat)
            sim_res = self.simulate_dynamic_npi_61(sim_mat, sim_res)

            sim_res = self.update_sim_result_daily(sim_res, sim_mat, pid_trans, daily_infections, pid_blocked, wg_close)
            sim_mat = self.simulate_imported(sim_mat, self.sim_date)
            self.Rt=self.calulate_Rt(sim_res)
            # features=self.get_situation_features(sim_mat,self.sim_date)
            # print(sim_mat["comp_this"][:100])
        sim_res = self.process_sim_result(sim_res)
        sim_res = self.update_sim_result_statistic(sim_res)
        self.sim_res = sim_res

        return sim_res

    def cal_p_overload_total(self, current_icu, cap_icu):
        """计算总的挤兑比例"""
        n_overload = sum(current_icu[current_icu > cap_icu] - cap_icu)
        p_overload_total = n_overload / sum(current_icu)  # 多少比例危重患者无法得到HealthCare

        return p_overload_total

    def cal_metrics(self):
        self.current_icu = self.sim_res["current_icu"]   # 危重症患者存量
        self.daily_onset = self.sim_res["daily_onset"]  # 每日新增发病
        self.daily_infect = self.sim_res["daily_infect"]

        self.r_overload_peak = max(max(self.current_icu) / (self.n_cap_icu*self.samp_rate) - 1, 0)  # 峰值过载率
        self.p_overload_total = self.cal_p_overload_total(self.current_icu, (self.n_cap_icu*self.samp_rate))  # 挤兑比例
        self.n_onset_peak = max(self.daily_onset)  # 峰值发病人数
        self.n_onset_total = sum(self.daily_onset)  # 总发病人数

        print("ICU峰值过载",round(self.r_overload_peak * 100,2))
        print("ICU总过载",round(self.p_overload_total * 100,2))
        print("high-level天数", self.sim_res["levels"].count(self.high_level ))
        print("ICU过载天数",np.sum(self.current_icu>(self.n_cap_icu*self.samp_rate)))
        print("感染峰值人数",max(self.daily_infect))
        print("感染峰值日",np.argmax(self.daily_infect))


if __name__=='__main__':
    ES = EpidemicSimulation(city_name="Shenzhen",
                            ptrans=0.07, uptake_scenario="Uptake 00",

                            is_npi_gather=False, npi_max_gather=10,
                            rate_iso_p_work_off=0,

                            p_mask=0.8,

                            init_level=1,
                            th1=np.inf, th2=np.inf,

                            p_drug=0.8,
                            is_samp=True, is_spatial=True, samp_rate=0.1,
                            duration_sigma=0.2, is_fast_r0=True,
                            max_iterday=150, n_imported_day0=100,n_imported_daily=1)
    start_time = time.time()
    res = ES.run()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time: ", execution_time)


    ES.cal_metrics()
    print('R0:', res['Rt'])
    plt.plot(res["daily_infect"], color="blue")
    # plt.plot(np.array(res["levels"]) * max(res["daily_infect"]) / 4, color="orangered")
    #
    # plt.show()
    #
    # plt.plot(res["current_icu"])
    # plt.hlines(ES.n_cap_icu * ES.samp_rate, 0, 180)
    # plt.hlines(ES.n_cap_icu * ES.samp_rate * ES.th1, 0, 180)
    # plt.hlines(ES.n_cap_icu * ES.samp_rate * ES.th2, 0, 180)
    # # plt.hlines(ES.n_cap_icu * ES.samp_rate * ES.th3, 0, 180)
    # # plt.hlines(ES.n_cap_icu * ES.samp_rate * ES.th4, 0, 180)
    # plt.plot(np.array(res["levels"]) * max(res["current_icu"]) / 4, color="orangered")
    # smoothed = np.convolve(res["current_icu"], np.ones(7) / 7, mode='same')
    # # plt.ylim(0, 100)
    # plt.plot(smoothed, color="green")
    # plt.show()
    #
    # smoothed = np.convolve(res["daily_infect"], np.ones(7) / 7, mode='same')
    # plt.plot(smoothed, color="blue")
    # plt.plot(np.array(res["levels"]) * max(res["daily_infect"]) / 4, color="orangered")
    # plt.show()
    # print(ES.city_name)
    # print(res['info'])
    #
    plt.show()
    #