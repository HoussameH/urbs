import math
import pyomo.core as pyomo
import numpy as np

BigM = 10000


def add_storage(m):
    # storage (e.g. hydrogen, pump storage)
    indexlist = set()
    for key in m.storage_dict["eff-in"]:
        indexlist.add(tuple(key)[2])
    m.sto = pyomo.Set(
        initialize=indexlist,
        doc='Set of storage technologies')

    # storage tuples
    m.sto_tuples = pyomo.Set(
        within=m.stf * m.sit * m.sto * m.com,
        initialize=tuple(m.storage_dict["eff-in"].keys()),
        doc='Combinations of possible storage by site,'
            'e.g. (2020,Mid,Bat,Elec)')

    # tuples for intertemporal operation
    if m.mode['int']:
        m.operational_sto_tuples = pyomo.Set(
            within=m.sit * m.sto * m.com * m.stf * m.stf,
            initialize=[(sit, sto, com, stf, stf_later)
                        for (sit, sto, com, stf, stf_later)
                        in op_sto_tuples(m.sto_tuples, m)],
            doc='Processes that are still operational through stf_later'
                '(and the relevant years following), if built in stf'
                'in stf.')
        m.inst_sto_tuples = pyomo.Set(
            within=m.sit * m.sto * m.com * m.stf,
            initialize=[(sit, sto, com, stf)
                        for (sit, sto, com, stf)
                        in inst_sto_tuples(m)],
            doc='Installed storages that are still operational through stf')

    # storage tuples for storages with fixed initial state
    m.sto_init_bound_tuples = pyomo.Set(
        within=m.stf * m.sit * m.sto * m.com,
        initialize=tuple(m.stor_init_bound_dict.keys()),
        doc='storages with fixed initial state')

    # storage tuples for storages with given energy to power ratio
    m.sto_ep_ratio_tuples = pyomo.Set(
        within=m.stf * m.sit * m.sto * m.com,
        initialize=tuple(m.sto_ep_ratio_dict.keys()),
        doc='storages with given energy to power ratio')

    # Variables
    m.cap_sto_c_new = pyomo.Var(
        m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='New storage size (MWh)')
    m.cap_sto_p_new = pyomo.Var(
        m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='New  storage power (MW)')

    # storage capacities as expression objects
    m.cap_sto_c = pyomo.Expression(
        m.sto_tuples,
        rule=def_storage_capacity_rule,
        doc='Total storage size (MWh)')
    m.cap_sto_p = pyomo.Expression(
        m.sto_tuples,
        rule=def_storage_power_rule,
        doc='Total storage power (MW)')

    m.e_sto_in = pyomo.Var(
        m.tm, m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow into storage (MW) per timestep')
    m.e_sto_out = pyomo.Var(
        m.tm, m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Power flow out of storage (MW) per timestep')
    m.e_sto_throughput = pyomo.Var(
        m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Energy throughput of storage (MWh) over time of analysis'
    )
    m.e_sto_charge_discharge_state = pyomo.Var(
        m.tm, m.sto_tuples,
        within=pyomo.Binary,
        doc='Indicates if the battery is charging(1) or discharging(0)'
    )
    # m.e_sto_discharge_state = pyomo.Var(
    #     m.tm, m.sto_tuples,
    #     within=pyomo.Binary,
    #     doc='Indicates if the battery is discharging'
    # )

    m.e_sto_con = pyomo.Var(
        m.t, m.sto_tuples,
        within=pyomo.NonNegativeReals,
        doc='Energy content of storage (MWh) in timestep')

    # storage rules
    m.def_storage_state = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=def_storage_state_rule,
        doc='storage[t] = (1 - sd) * storage[t-1] + in * eff_i - out / eff_o')

    # m.storage_charge_discharge_cancelation = pyomo.Constraint(
    #     m.tm, m.sto_tuples,
    #     rule=def_storage_charge_discharge_cancellation_rule,
    #     doc='e_charge_state + e_discharge_state <= 1')

    m.def_throughput_state = pyomo.Constraint(
        m.sto_tuples,
        rule=def_throughput_state_rule,
        doc='throughput = throughput + out / eff-out')

    m.def_limit_cycle = pyomo.Constraint(
        m.sto_tuples,
        rule=def_limit_cyle_rule,
        doc='capacity <= throughput / maximum cycles (for studied period)'
    )

    m.res_storage_input_by_power = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=res_storage_input_by_power_rule,
        doc='storage input <= storage power')

    m.res_storage_charge_input_by_power = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=res_storage_charge_input_by_power_rule,
        doc='P in <= charge state * BigM'
    )

    m.res_storage_output_by_power = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=res_storage_output_by_power_rule,
        doc='storage output <= storage power')

    m.res_storage_discharge_output_by_power = pyomo.Constraint(
        m.tm, m.sto_tuples,
        rule=res_storage_discharge_output_by_power_rule,
        doc='P out <= (1- charge state) * BigM'
    )

    m.res_storage_state_by_capacity = pyomo.Constraint(
        m.t, m.sto_tuples,
        rule=res_storage_state_by_capacity_rule,
        doc='storage content <= storage capacity')
    m.res_storage_power = pyomo.Constraint(
        m.sto_tuples,
        rule=res_storage_power_rule,
        doc='storage.cap-lo-p <= storage power <= storage.cap-up-p')
    m.res_storage_capacity = pyomo.Constraint(
        m.sto_tuples,
        rule=res_storage_capacity_rule,
        doc='storage.cap-lo-c <= storage capacity <= storage.cap-up-c')
    m.res_initial_and_final_storage_state = pyomo.Constraint(
        m.t, m.sto_init_bound_tuples,
        rule=res_initial_and_final_storage_state_rule,
        doc='storage content initial == and final >= storage.init * capacity')
    m.res_initial_and_final_storage_state_var = pyomo.Constraint(
        m.t, m.sto_tuples - m.sto_init_bound_tuples,
        rule=res_initial_and_final_storage_state_var_rule,
        doc='storage content initial <= final, both variable')
    m.def_storage_energy_power_ratio = pyomo.Constraint(
        m.sto_ep_ratio_tuples,
        rule=def_storage_energy_power_ratio_rule,
        doc='storage capacity = storage power * storage E2P ratio')

    return m


# constraints

# storage content in timestep [t] == storage content[t-1] * (1-discharge)
# + newly stored energy * input efficiency
# - retrieved energy / output efficiency
def def_storage_state_rule(m, t, stf, sit, sto, com):
    return (m.e_sto_con[t, stf, sit, sto, com] ==
            m.e_sto_con[t - 1, stf, sit, sto, com] *
            (1 - m.storage_dict['discharge']
            [(stf, sit, sto, com)]) ** m.dt.value +
            m.e_sto_in[t, stf, sit, sto, com] *
            m.storage_dict['eff-in'][(stf, sit, sto, com)] -
            m.e_sto_out[t, stf, sit, sto, com] /
            m.storage_dict['eff-out'][(stf, sit, sto, com)])


# # Makes sure that the storage is not charged and discharged at the same time
# # Charge state + discharge state <=1
#
# def def_storage_charge_discharge_cancellation_rule(m, t, stf, sit, sto, com):
#     return m.e_sto_charge_state[t, stf, sit, sto, com] <= 1 - m.e_sto_discharge_state[t, stf, sit, sto, com]


# Energy throughput of the battery
# throughput = Sum(retrieved energy[t]/output efficiency) over tm

def def_throughput_state_rule(m, stf, sit, sto, com):
    throughput = 0
    for t in m.tm:
        throughput = throughput + m.e_sto_out[t, stf, sit, sto, com] / m.storage_dict['eff-out'][(stf, sit, sto, com)]

    return m.e_sto_throughput[stf, sit, sto, com] == throughput


# Limit the number of cycles of the storage system

def def_limit_cyle_rule(m, stf, sit, sto, com):
    periodlen = len(m.tm) + 1

    maxcycles = m.storage_dict['cycles'][(stf, sit, sto, com)]
    if np.isnan(maxcycles):
        return pyomo.Constraint.Feasible
    depre = m.storage_dict['depreciation'][(stf, sit, sto, com)]

    cycles = (maxcycles / (depre * 8760)) * periodlen

    if np.isnan(m.storage_dict['cyclecoef'][(stf, sit, sto, com)]) \
            or m.storage_dict['cyclecoef'][(stf, sit, sto, com)] == 0:
        return m.cap_sto_c[stf, sit, sto, com] <= m.e_sto_throughput[stf, sit, sto, com] / cycles
    else:
        return m.cap_sto_c[stf, sit, sto, com] \
               <= (m.e_sto_throughput[stf, sit, sto, com] * m.storage_dict['cyclecoef'][(stf, sit, sto, com)]) / cycles


# storage capacity (for m.cap_sto_c expression)


def def_storage_capacity_rule(m, stf, sit, sto, com):
    if m.mode['int']:
        if (sit, sto, com, stf) in m.inst_sto_tuples:
            if (min(m.stf), sit, sto, com) in m.sto_const_cap_c_dict:
                cap_sto_c = m.storage_dict['cap-up-c'][(stf, sit, sto, com)]
            else:
                cap_sto_c = (
                        sum(m.cap_sto_c_new[stf_built, sit, sto, com]
                            for stf_built in m.stf
                            if (sit, sto, com, stf_built, stf) in
                            m.operational_sto_tuples) +
                        m.storage_dict['inst-cap-c'][(min(m.stf), sit, sto, com)])
        else:
            cap_sto_c = (
                sum(m.cap_sto_c_new[stf_built, sit, sto, com]
                    for stf_built in m.stf
                    if (sit, sto, com, stf_built, stf) in
                    m.operational_sto_tuples))
    else:
        if (stf, sit, sto, com) in m.sto_const_cap_c_dict:
            cap_sto_c = m.storage_dict['inst-cap-c'][(stf, sit, sto, com)]
        else:
            cap_sto_c = (m.cap_sto_c_new[stf, sit, sto, com] +
                         m.storage_dict['inst-cap-c'][(stf, sit, sto, com)])

    return cap_sto_c


# storage power (for m.cap_sto_p expression)


def def_storage_power_rule(m, stf, sit, sto, com):
    if m.mode['int']:
        if (sit, sto, com, stf) in m.inst_sto_tuples:
            if (min(m.stf), sit, sto, com) in m.sto_const_cap_p_dict:
                cap_sto_p = m.storage_dict['inst-cap-p'][
                    (stf, sit, sto, com)]
            else:
                cap_sto_p = (
                        sum(m.cap_sto_p_new[stf_built, sit, sto, com]
                            for stf_built in m.stf
                            if (sit, sto, com, stf_built, stf) in
                            m.operational_sto_tuples) +
                        m.storage_dict['inst-cap-p'][(min(m.stf), sit, sto, com)])
        else:
            cap_sto_p = (
                sum(m.cap_sto_p_new[stf_built, sit, sto, com]
                    for stf_built in m.stf
                    if (sit, sto, com, stf_built, stf)
                    in m.operational_sto_tuples))
    else:
        if (stf, sit, sto, com) in m.sto_const_cap_p_dict:
            cap_sto_p = m.storage_dict['inst-cap-p'][(stf, sit, sto, com)]
        else:
            cap_sto_p = (m.cap_sto_p_new[stf, sit, sto, com] +
                         m.storage_dict['inst-cap-p'][(stf, sit, sto, com)])

    return cap_sto_p


# storage input <= storage power
def res_storage_input_by_power_rule(m, t, stf, sit, sto, com):
    return (m.e_sto_in[t, stf, sit, sto, com] <= m.dt *
            m.cap_sto_p[stf, sit, sto, com])


# storage input <= charging state * BigM
def res_storage_charge_input_by_power_rule(m, tm, stf, sit, sto, com):
    cap_up = m.storage_dict['cap-up-p'][(stf, sit, sto, com)]
    if ~np.isinf(cap_up):
        maxcap = cap_up
    else:
        maxcap = BigM
    return m.e_sto_in[tm, stf, sit, sto, com] <= m.e_sto_charge_discharge_state[tm, stf, sit, sto, com] * maxcap


# storage output <= storage power
def res_storage_output_by_power_rule(m, t, stf, sit, sto, co):
    return (m.e_sto_out[t, stf, sit, sto, co] <= m.dt *
            m.cap_sto_p[stf, sit, sto, co])


# storage input <= charging state * BigM
def res_storage_discharge_output_by_power_rule(m, tm, stf, sit, sto, com):
    cap_up = m.storage_dict['cap-up-p'][(stf, sit, sto, com)]
    if ~np.isinf(cap_up):
        maxcap = cap_up
    else:
        maxcap = BigM
    return m.e_sto_out[tm, stf, sit, sto, com] <= (1 - m.e_sto_charge_discharge_state[tm, stf, sit, sto, com]) * maxcap


# storage content <= storage capacity
def res_storage_state_by_capacity_rule(m, t, stf, sit, sto, com):
    return (m.e_sto_con[t, stf, sit, sto, com] <=
            m.cap_sto_c[stf, sit, sto, com])


# lower bound <= storage power <= upper bound
def res_storage_power_rule(m, stf, sit, sto, com):
    return (m.storage_dict['cap-lo-p'][(stf, sit, sto, com)],
            m.cap_sto_p[stf, sit, sto, com],
            m.storage_dict['cap-up-p'][(stf, sit, sto, com)])


# lower bound <= storage capacity <= upper bound
def res_storage_capacity_rule(m, stf, sit, sto, com):
    return (m.storage_dict['cap-lo-c'][(stf, sit, sto, com)],
            m.cap_sto_c[stf, sit, sto, com],
            m.storage_dict['cap-up-c'][(stf, sit, sto, com)])


# initialization of storage content in first timestep t[1]
# forced minimun  storage content in final timestep t[len(m.t)]
# content[t=1] == storage capacity * fraction <= content[t=final]
def res_initial_and_final_storage_state_rule(m, t, stf, sit, sto, com):
    if t == m.t[1]:  # first timestep (Pyomo uses 1-based indexing)
        return (m.e_sto_con[t, stf, sit, sto, com] ==
                m.cap_sto_c[stf, sit, sto, com] *
                m.storage_dict['init'][(stf, sit, sto, com)])
    elif t == m.t[len(m.t)]:  # last timestep
        return (m.e_sto_con[t, stf, sit, sto, com] >=
                m.cap_sto_c[stf, sit, sto, com] *
                m.storage_dict['init'][(stf, sit, sto, com)])
    else:
        return pyomo.Constraint.Skip


def res_initial_and_final_storage_state_var_rule(m, t, stf, sit, sto, com):
    return (m.e_sto_con[m.t[1], stf, sit, sto, com] <=
            m.e_sto_con[m.t[len(m.t)], stf, sit, sto, com])


def def_storage_energy_power_ratio_rule(m, stf, sit, sto, com):
    return (m.cap_sto_c[sit, sto, com] == m.cap_sto_p[sit, sto, com] *
            m.storage_dict['ep-ratio'][(sit, sto, com)])


# storage balance
def storage_balance(m, tm, stf, sit, com):
    """callesd in commodity balance
    For a given commodity co and timestep tm, calculate the balance of
    storage input and output """

    return sum(m.e_sto_in[(tm, stframe, site, storage, com)] -
               m.e_sto_out[(tm, stframe, site, storage, com)]
               # usage as input for storage increases consumption
               # output from storage decreases consumption
               for stframe, site, storage, commodity in m.sto_tuples
               if site == sit and stframe == stf and commodity == com)


# storage costs


def storage_cost(m, cost_type):
    """returns storage cost function for the different cost types"""
    if cost_type == 'Invest':
        cost = sum(m.cap_sto_p_new[s] *
                   m.storage_dict['inv-cost-p'][s] *
                   m.storage_dict['invcost-factor'][s] +
                   m.cap_sto_c_new[s] *
                   m.storage_dict['inv-cost-c'][s] *
                   m.storage_dict['invcost-factor'][s]
                   for s in m.sto_tuples)
        if m.mode['int']:
            cost -= sum(m.cap_sto_p_new[s] *
                        m.storage_dict['inv-cost-p'][s] *
                        m.storage_dict['overpay-factor'][s] +
                        m.cap_sto_c_new[s] *
                        m.storage_dict['inv-cost-c'][s] *
                        m.storage_dict['overpay-factor'][s]
                        for s in m.sto_tuples)
        return cost
    elif cost_type == 'Fixed':
        return sum((m.cap_sto_p[s] * m.storage_dict['fix-cost-p'][s] +
                    m.cap_sto_c[s] * m.storage_dict['fix-cost-c'][s]) *
                   m.storage_dict['cost_factor'][s]
                   for s in m.sto_tuples)
    elif cost_type == 'Variable':
        return sum(m.e_sto_con[(tm,) + s] * m.weight *
                   m.storage_dict['var-cost-c'][s] *
                   m.storage_dict['cost_factor'][s] +
                   (m.e_sto_in[(tm,) + s] + m.e_sto_out[(tm,) + s]) *
                   m.weight * m.storage_dict['var-cost-p'][s] *
                   m.storage_dict['cost_factor'][s]
                   for tm in m.tm
                   for s in m.sto_tuples)


def op_sto_tuples(sto_tuple, m):
    """ s.a. op_pro_tuples
    """
    op_sto = []
    sorted_stf = sorted(list(m.stf))

    for (stf, sit, sto, com) in sto_tuple:
        for stf_later in sorted_stf:
            index_helper = sorted_stf.index(stf_later)
            if stf_later == max(sorted_stf):
                if (stf_later +
                        m.global_prop_dict['value'][(max(sorted_stf), 'Weight')] -
                        1 <= stf +
                        m.storage_dict['depreciation'][(stf, sit, sto, com)]):
                    op_sto.append((sit, sto, com, stf, stf_later))
            elif (sorted_stf[index_helper + 1] <=
                  stf +
                  m.storage_dict['depreciation'][(stf, sit, sto, com)] and
                  stf <= stf_later):
                op_sto.append((sit, sto, com, stf, stf_later))
            else:
                pass

    return op_sto


def inst_sto_tuples(m):
    """ s.a. inst_pro_tuples
    """
    inst_sto = []
    sorted_stf = sorted(list(m.stf))

    for (stf, sit, sto, com) in m.inst_sto.index:
        for stf_later in sorted_stf:
            index_helper = sorted_stf.index(stf_later)
            if stf_later == max(m.stf):
                if (stf_later +
                        m.global_prop_dict['value'][(max(sorted_stf), 'Weight')] -
                        1 < min(m.stf) +
                        m.storage_dict['lifetime'][(stf, sit, sto, com)]):
                    inst_sto.append((sit, sto, com, stf_later))
            elif (sorted_stf[index_helper + 1] <=
                  min(m.stf) + m.storage_dict['lifetime'][
                      (stf, sit, sto, com)]):
                inst_sto.append((sit, sto, com, stf_later))

    return inst_sto
