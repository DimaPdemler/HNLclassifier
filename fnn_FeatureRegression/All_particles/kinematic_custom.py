from numpy import array, zeros_like, zeros, sign, sqrt, cos, sin, arctan, exp, pi
import numpy as np
import random
from vector import array as v_array
from vector import arr
import vector



def eta_to_theta(eta):
    return 2 * arctan(exp(-eta))


def vec_3D(norm, phi, theta, hep_form=True):
    np_output = array(
        [norm * sin(theta) * cos(phi), norm * sin(theta) * sin(phi), norm * cos(theta)]
    )
    if hep_form:
        output = arr({"x": np_output[0], "y": np_output[1], "z": np_output[2]})
        return output
    return array


def deltaphi(phi1, phi2):
    """
    Arguments:
        -phi1 : azimuthal angle of the first particle
        -phi2 : azimuthal angle of the second particle
    """

    return abs((((phi1 - phi2) + pi) % (2 * pi)) - pi)


def deltaphi3(
    pt_1,
    pt_2,
    pt_3,
    pt_MET,
    phi_1,
    phi_2,
    phi_3,
    phi_MET,
    eta_1,
    eta_2,
    eta_3,
    mass_1,
    mass_2,
    mass_3,
):
    """
    Arguments:
        -pt_1,2,3 : transverse momentum of the leptons
        -pt_MET : norm of missing transverse momentum
        -phi_1,2,3 : azimuthal angle of the leptons
        -phi_MET : azimuthal angle of the missing transverse momentum
        -eta_1,2,3 : pseudorapidity of the leptons
        -eta_1,2,3 : mass of the leptons
    Output:
        -All combinations of deltaphi between 1 object and the sum of two others
    """
    n = len(pt_1)
    eta_MET = []
    mass_MET = []
    if type(pt_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(pt_1)
        mass_MET = zeros_like(pt_1)
    vector_MET = arr({"pt": pt_MET, "phi": phi_MET, "eta": eta_MET, "M": mass_MET})
    vector_1 = arr({"pt": pt_1, "phi": phi_1, "eta": eta_1, "M": mass_1})
    vector_2 = arr({"pt": pt_2, "phi": phi_2, "eta": eta_2, "M": mass_2})
    vector_3 = arr({"pt": pt_3, "phi": phi_3, "eta": eta_3, "M": mass_3})
    vectors = [vector_1, vector_2, vector_3, vector_MET]

    groups = [
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [3, 0, 1],
        [3, 0, 2],
        [3, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 0, 3],
        [1, 2, 3],
        [2, 0, 3],
        [2, 1, 3],
    ]
    deltaphis = []

    for comb in groups:
        deltaphis.append(
            deltaphi(vectors[comb[0]].phi, (vectors[comb[1]] + vectors[comb[2]]).phi)
        )

    return deltaphis


def deltaeta(eta1, eta2):
    """
    Arguments:
        -eta1 : pseudorapidity of the first particle
        -eta2 : pseudorapidity of the second particle
    """

    return abs(eta1 - eta2)


def deltaeta3(
    pt_1, pt_2, pt_3, phi_1, phi_2, phi_3, eta_1, eta_2, eta_3, mass_1, mass_2, mass_3
):
    """
    Arguments:
        -pt_1,2,3 : transverse momentum of the leptons
        -phi_1,2,3 : azimuthal angle of the leptons
        -eta_1,2,3 : pseudorapidity of the leptons
        -eta_1,2,3 : mass of the leptons
    Output:
        -All combinations of deltaeta between 1 lepton and the sum of two others
    """
    vector_1 = arr({"pt": pt_1, "phi": phi_1, "eta": eta_1, "M": mass_1})
    vector_2 = arr({"pt": pt_2, "phi": phi_2, "eta": eta_2, "M": mass_2})
    vector_3 = arr({"pt": pt_3, "phi": phi_3, "eta": eta_3, "M": mass_3})
    vectors = [vector_1, vector_2, vector_3]

    groups = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
    deltaetas = []

    for comb in groups:
        deltaetas.append(
            deltaeta(vectors[comb[0]].eta, (vectors[comb[1]] + vectors[comb[2]]).eta)
        )

    return deltaetas


def deltaR(eta1, eta2, phi1, phi2):
    """
    Arguments:
        -eta1 : pseudorapidity of the first particle
        -eta2 : pseudorapidity of the second particle
        -phi1 : azimuthal angle of the first particle
        -phi2 : azimuthal angle of the second particle
    """

    return sqrt(deltaeta(eta1, eta2) ** 2 + deltaphi(phi1, phi2) ** 2)


def deltaR3(
    pt_1, pt_2, pt_3, phi_1, phi_2, phi_3, eta_1, eta_2, eta_3, mass_1, mass_2, mass_3
):
    """
    Arguments:
        -pt_1,2,3 : transverse momentum of the leptons
        -phi_1,2,3 : azimuthal angle of the leptons
        -eta_1,2,3 : pseudorapidity of the leptons
        -eta_1,2,3 : mass of the leptons
    Output:
        -All combinations of deltaR between 1 lepton and the sum of two others
    """
    vector_1 = arr({"pt": pt_1, "phi": phi_1, "eta": eta_1, "M": mass_1})
    vector_2 = arr({"pt": pt_2, "phi": phi_2, "eta": eta_2, "M": mass_2})
    vector_3 = arr({"pt": pt_3, "phi": phi_3, "eta": eta_3, "M": mass_3})
    vectors = [vector_1, vector_2, vector_3]

    groups = [[0, 1, 2], [1, 0, 2], [2, 0, 1]]
    deltaRs = []

    for comb in groups:
        vec_sum = vectors[comb[1]] + vectors[comb[2]]
        deltaRs.append(
            deltaR(vectors[comb[0]].eta, vec_sum.eta, vectors[comb[0]].phi, vec_sum.phi)
        )

    return deltaRs


def sum_pt(pts, phis, etas, masses):
    """
    Aguments :
        -pts : transverse momentum of the particles
        -phis : azimuthal angles of the particles
        -etas : pseudorapidity of the particles
        -masses : masses of the particles
    All arguments have 2 coordinates :
        -the first component corresponds to the type of particle (muon, tau, MET)
        -The second coordinate corresponds to the event.
    Output :
        -Sum of the transverse momentum of the three leptons and the MET
    """

    p_tot = arr({"pt": pts[0], "phi": phis[0], "eta": etas[0], "M": masses[0]})
    for i in range(1, len(masses)):
        p_tot += arr({"pt": pts[i], "phi": phis[i], "eta": etas[i], "M": masses[i]})
    return p_tot.pt


def transverse_mass(pt_1, pt_2, phi_1, phi_2):
    """
    Arguments :
        -pt_1 : transverse momentum of the first particle
        -pt_2 : transverse momentum of the second particle
        -phi_1 : azimuthal angle of the first particle
        -phi_2 : azimuthal angle of the second particle
    """
    
    result = 2.0 * pt_1 * pt_2 * (1.0 - np.cos(phi_1 - phi_2))

    # mask= result <0
    # mask_result=result[mask]
    # if len(mask_result)>0:
    #     print("mask_result", mask_result)
    
    # if result <0:
    #     print("pt1, pt2, phi1, phi2", pt_1, pt_2, phi_1, phi_2)
    # result[result < 0] = 0.0
    return np.sqrt(result)



def transverse_mass3(
    pt_1,
    pt_2,
    pt_3,
    pt_MET,
    phi_1,
    phi_2,
    phi_3,
    phi_MET,
    eta_1,
    eta_2,
    eta_3,
    mass_1,
    mass_2,
    mass_3,
):
    """
    Arguments:
        -pt_1,2,3 : transverse momentum of the leptons
        -pt_MET : norm of missing transverse momentum
        -phi_1,2,3 : azimuthal angle of the leptons
        -phi_MET : azimuthal angle of the missing transverse momentum
        -eta_1,2,3 : pseudorapidity of the leptons
        -eta_1,2,3 : mass of the leptons
    Output:
        -All combinations of transverse masses between 1 object and the sum of two others
    """
    n = len(pt_1)
    eta_MET = []
    mass_MET = []
    if type(pt_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(pt_1)
        mass_MET = zeros_like(pt_1)
    vector_MET = arr({"pt": pt_MET, "phi": phi_MET, "eta": eta_MET, "M": mass_MET})
    vector_1 = arr({"pt": pt_1, "phi": phi_1, "eta": eta_1, "M": mass_1})
    vector_2 = arr({"pt": pt_2, "phi": phi_2, "eta": eta_2, "M": mass_2})
    vector_3 = arr({"pt": pt_3, "phi": phi_3, "eta": eta_3, "M": mass_3})
    vectors = [vector_1, vector_2, vector_3, vector_MET]

    groups = [
        [0, 1, 2],
        [1, 0, 2],
        [2, 0, 1],
        [3, 0, 1],
        [3, 0, 2],
        [3, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 0, 3],
        [1, 2, 3],
        [2, 0, 3],
        [2, 1, 3],
    ]
    transverse_masses = []

    for comb in groups:
        vec_sum = vectors[comb[0]] + vectors[comb[1]]
        transverse_masses.append(
            transverse_mass(
                vectors[comb[0]].pt, vec_sum.pt, vectors[comb[0]].phi, vec_sum.phi
            )
        )

    return transverse_masses


def total_transverse_mass(pt_1, pt_2, pt_3, pt_miss, phi_1, phi_2, phi_3, phi_miss):
    """
    Arguments :
        -pt_1 : transverse momentum of the first particle
        -pt_2 : transverse momentum of the second particle
        -pt_3 : transverse momentum of the third particle
        -pt_miss : missing transverse momentum
        -phi_1 : azimuthal angle of the first particle
        -phi_2 : azimuthal angle of the second particle
        -phi_3 : azimuthal angle of the third particle
        -phi_miss : azimuthal angle of missing particles
    """

    return sqrt(
        transverse_mass(pt_1, pt_2, phi_1, phi_2) ** 2
        + transverse_mass(pt_1, pt_3, phi_1, phi_3) ** 2
        + transverse_mass(pt_2, pt_3, phi_2, phi_3) ** 2
        + transverse_mass(pt_1, pt_miss, phi_1, phi_miss) ** 2
        + transverse_mass(pt_2, pt_miss, phi_2, phi_miss) ** 2
        + transverse_mass(pt_3, pt_miss, phi_3, phi_miss) ** 2
    )


def invariant_mass(pts, phis, etas, masses):
    """
    Aguments :
        -pts : transverse momentum of the particles
        -phis : azimuthal angles of the particles
        -etas : pseudorapidity of the particles
        -masses : masses of the particles
    All arguments have 2 coordinates :
        -the first component corresponds to the type of particle (muon, tau, MET)
        -The second coordinate corresponds to the event.
    Output : invariant mass of the sum of the 4-vectors of all objects passed to the function
    """
    p_tot = arr({"pt": pts[0], "phi": phis[0], "eta": etas[0], "M": masses[0]})
    for i in range(1, len(masses)):
        p_tot += arr({"pt": pts[i], "phi": phis[i], "eta": etas[i], "M": masses[i]})
    return p_tot.mass






def HNL_CM_angles_with_MET(
    pt_1, pt_2, pt_3, pt_MET,
    phi_1, phi_2, phi_3, phi_MET,
    eta_1, eta_2, eta_3,
    mass_1, mass_2, mass_3,
):
    """
    Arguments :
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[angle1, angle2, angle3] : angles between the 3 possible lepton pairs in their rest frame.
    """
    
    n = len(pt_1)

    if type(pt_1) == list:
        eta_MET = [0] * n
        mass_MET = [0] * n
    else:
        eta_MET = zeros_like(pt_1)
        mass_MET = zeros_like(pt_1)

    vector_MET = arr({"pt": pt_MET, "phi": phi_MET, "eta": eta_MET, "M": mass_MET})

    # Lepton properties in dictionaries for easier looping
    pts = {1: pt_1, 2: pt_2, 3: pt_3}
    phis = {1: phi_1, 2: phi_2, 3: phi_3}
    etas = {1: eta_1, 2: eta_2, 3: eta_3}
    masses = {1: mass_1, 2: mass_2, 3: mass_3}

    # All possible pairs
    pair_candidate = [[1, 2], [1, 3], [2, 3]]
    angles = []

    for pair in pair_candidate:
        i, j = pair
        vector_i = arr({"pt": pts[i], "phi": phis[i], "eta": etas[i], "M": masses[i]})
        vector_j = arr({"pt": pts[j], "phi": phis[j], "eta": etas[j], "M": masses[j]})

        vector_tot = vector_i + vector_j + vector_MET
        vector_i = vector_i.boostCM_of_p4(vector_tot)
        vector_j = vector_j.boostCM_of_p4(vector_tot)

        angle = vector_i.deltaangle(vector_j)
        angles.append(angle)

    return angles


def W_CM_angles_to_plane(
    pt_1, pt_2, pt_3,
    phi_1, phi_2, phi_3,
    eta_1, eta_2, eta_3,
    mass_1, mass_2, mass_3,
):
    """
    Arguments :
        -pt_1,2,3 : transverse momentum of the three leptons
        -phi_1,2,3 : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[angle1, angle2, angle3] : angles between 1 lepton and the plane formed by the 2 other leptons in the rest frame 
                                    of the 3 leptons. There are 3 possible lepton pairs -> 3 angles in the output.
    """
    
    # Lepton properties in dictionaries for easier looping
    pts = {1: pt_1, 2: pt_2, 3: pt_3}
    phis = {1: phi_1, 2: phi_2, 3: phi_3}
    etas = {1: eta_1, 2: eta_2, 3: eta_3}
    masses = {1: mass_1, 2: mass_2, 3: mass_3}

    # All possible pairs
    pair_candidate = [[1, 2], [1, 3], [2, 3]]
    angles = []

    for pair in pair_candidate:
        i, j = pair
        k = next(value for value in [1, 2, 3] if value != i and value != j)

        vector_first = arr({"pt": pts[k], "phi": phis[k], "eta": etas[k], "M": masses[k]})
        vector_i = arr({"pt": pts[i], "phi": phis[i], "eta": etas[i], "M": masses[i]})
        vector_j = arr({"pt": pts[j], "phi": phis[j], "eta": etas[j], "M": masses[j]})
        
        vector_tot = vector_i + vector_j + vector_first
        vector_i = vector_i.boostCM_of_p4(vector_tot)
        vector_j = vector_j.boostCM_of_p4(vector_tot)
        vector_first = vector_first.boostCM_of_p4(vector_tot)
        
        normal = vector_i.cross(vector_j)
        angle = vector_first.deltaangle(normal)
        
        angles.append(abs(pi / 2 - angle))
    return angles



def W_CM_angles_to_plane_with_MET(
    pt_1, pt_2, pt_3, pt_MET,
    phi_1, phi_2, phi_3, phi_MET,
    eta_1, eta_2, eta_3,
    mass_1, mass_2, mass_3,
):
    """
    Arguments :
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[angle1, angle2, angle3] : angles between 1 lepton and the plane formed by the 2 other leptons in the rest frame 
                                    of the 3 leptons. There are 3 possible lepton pairs -> 3 angles in the output.
    """
    
    n = len(pt_1)
    if type(pt_1) == list:
        eta_MET = [0] * n
        mass_MET = [0] * n
    else:
        eta_MET = zeros_like(pt_1)
        mass_MET = zeros_like(pt_1)
    
    vector_MET = arr({"pt": pt_MET, "phi": phi_MET, "eta": eta_MET, "M": mass_MET})

    # Lepton properties in dictionaries for easier looping
    pts = {1: pt_1, 2: pt_2, 3: pt_3}
    phis = {1: phi_1, 2: phi_2, 3: phi_3}
    etas = {1: eta_1, 2: eta_2, 3: eta_3}
    masses = {1: mass_1, 2: mass_2, 3: mass_3}

    # All possible pairs
    pair_candidate = [[1, 2], [1, 3], [2, 3]]
    angles = []

    for pair in pair_candidate:
        i, j = pair
        k = next(value for value in [1, 2, 3] if value != i and value != j)

        vector_first = arr({"pt": pts[k], "phi": phis[k], "eta": etas[k], "M": masses[k]})
        vector_i = arr({"pt": pts[i], "phi": phis[i], "eta": etas[i], "M": masses[i]})
        vector_j = arr({"pt": pts[j], "phi": phis[j], "eta": etas[j], "M": masses[j]})
        
        vector_tot = vector_i + vector_j + vector_first + vector_MET
        vector_i = vector_i.boostCM_of_p4(vector_tot)
        vector_j = vector_j.boostCM_of_p4(vector_tot)
        vector_first = vector_first.boostCM_of_p4(vector_tot)
        
        normal = vector_i.cross(vector_j)
        angle = vector_first.deltaangle(normal)
        
        angles.append(abs(pi / 2 - angle))
    return angles


def W_CM_angles(
    pt_1,
    pt_2,
    pt_3,
    pt_MET,
    phi_1,
    phi_2,
    phi_3,
    phi_MET,
    eta_1,
    eta_2,
    eta_3,
    mass_1,
    mass_2,
    mass_3,
):
    """
    Arguments :
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -Angle between the momenta of 2 objects in the center of mass frame of the first W boson (without considering missing momentum),
         for all 6 combination of 2 objects
    """

    n = len(pt_1)
    eta_MET = []
    mass_MET = []
    if type(pt_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(pt_1)
        mass_MET = zeros_like(pt_1)
    vector_MET = arr({"pt": pt_MET, "phi": phi_MET, "eta": eta_MET, "M": mass_MET})
    vector_1 = arr({"pt": pt_1, "phi": phi_1, "eta": eta_1, "M": mass_1})
    vector_2 = arr({"pt": pt_2, "phi": phi_2, "eta": eta_2, "M": mass_2})
    vector_3 = arr({"pt": pt_3, "phi": phi_3, "eta": eta_3, "M": mass_3})

    vectors = [vector_1, vector_2, vector_3, vector_MET]

    vector_tot = vector_1 + vector_2 + vector_3

    for i in range(len(vectors)):
        vectors[i] = vectors[i].boostCM_of_p4(vector_tot)

    pairs = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]]
    angles = []

    for pair in pairs:
        angle = vectors[pair[0]].deltaangle(vectors[pair[1]])
        angles.append(angle)

    return angles


def W_CM_angles_with_MET(
    pt_1,
    pt_2,
    pt_3,
    pt_MET,
    phi_1,
    phi_2,
    phi_3,
    phi_MET,
    eta_1,
    eta_2,
    eta_3,
    mass_1,
    mass_2,
    mass_3,
):
    """
    Arguments :
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -Angle between the momenta of 2 objects in the center of mass frame of the first W boson (with missing momentum),
         for all 6 combination of 2 objects
    """

    n = len(pt_1)
    eta_MET = []
    mass_MET = []
    if type(pt_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(pt_1)
        mass_MET = zeros_like(pt_1)
    vector_MET = arr({"pt": pt_MET, "phi": phi_MET, "eta": eta_MET, "M": mass_MET})
    vector_1 = arr({"pt": pt_1, "phi": phi_1, "eta": eta_1, "M": mass_1})
    vector_2 = arr({"pt": pt_2, "phi": phi_2, "eta": eta_2, "M": mass_2})
    vector_3 = arr({"pt": pt_3, "phi": phi_3, "eta": eta_3, "M": mass_3})

    vectors = [vector_1, vector_2, vector_3, vector_MET]

    vector_tot = vector_1 + vector_2 + vector_3 + vector_MET

    for i in range(len(vectors)):
        vectors[i] = vectors[i].boostCM_of_p4(vector_tot)

    pairs = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]]
    angles = []

    for pair in pairs:
        angle = vectors[pair[0]].deltaangle(vectors[pair[1]])
        angles.append(angle)

    return angles


def HNL_CM_masses(
    pt_1, pt_2, pt_3,
    phi_1, phi_2, phi_3,
    eta_1, eta_2, eta_3,
    mass_1, mass_2, mass_3,
):
    """
    Arguments :
        -pt_1,2,3 : transverse momentum of the three leptons
        -phi_1,2,3 : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[HNL_mass1, HNL_mass2, HNL_mass3] : invariant mass of the sum of 2 leptons. 
                    There are 3 possible lepton pairs -> 3 masses in the output.
    """

    # Lepton properties in dictionaries for easier looping
    pts = {1: pt_1, 2: pt_2, 3: pt_3}
    phis = {1: phi_1, 2: phi_2, 3: phi_3}
    etas = {1: eta_1, 2: eta_2, 3: eta_3}
    masses = {1: mass_1, 2: mass_2, 3: mass_3}

    # All possible pairs
    pair_candidate = [[1, 2], [1, 3], [2, 3]]
    HNL_masses = []

    for pair in pair_candidate:
        i, j = pair

        vector_i = arr({"pt": pts[i], "phi": phis[i], "eta": etas[i], "M": masses[i]})
        vector_j = arr({"pt": pts[j], "phi": phis[j], "eta": etas[j], "M": masses[j]})
        
        vector_tot = vector_i + vector_j
        HNL_masses.append(vector_tot.mass)
    return HNL_masses



def HNL_CM_masses_with_MET(
    pt_1, pt_2, pt_3,
    pt_MET,
    phi_1, phi_2, phi_3,
    phi_MET,
    eta_1, eta_2, eta_3,
    mass_1, mass_2, mass_3,
):
    """
    Arguments :
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[HNL_mass1, HNL_mass2, HNL_mass3] : invariant mass of the sum of 2 leptons. 
                    There are 3 possible lepton pairs -> 3 masses in the output.
    """

    # MET properties
    n = len(pt_1)
    eta_MET = [0] * n
    mass_MET = [0] * n
    vector_MET = arr({"pt": pt_MET, "phi": phi_MET, "eta": eta_MET, "M": mass_MET})

    # Lepton properties in dictionaries for easier looping
    pts = {1: pt_1, 2: pt_2, 3: pt_3}
    phis = {1: phi_1, 2: phi_2, 3: phi_3}
    etas = {1: eta_1, 2: eta_2, 3: eta_3}
    masses = {1: mass_1, 2: mass_2, 3: mass_3}

    # All possible pairs
    pair_candidate = [[1, 2], [1, 3], [2, 3]]
    HNL_masses = []

    for pair in pair_candidate:
        i, j = pair

        vector_i = arr({"pt": pts[i], "phi": phis[i], "eta": etas[i], "M": masses[i]})
        vector_j = arr({"pt": pts[j], "phi": phis[j], "eta": etas[j], "M": masses[j]})
        
        vector_tot = vector_i + vector_j + vector_MET
        HNL_masses.append(vector_tot.mass)
    return HNL_masses

def p4calc(
    pt_1, pt_2, pt_3,
    phi_1, phi_2, phi_3,
    eta_1, eta_2, eta_3,
    mass_1, mass_2, mass_3
):
    """
    Arguments :
        -pt_1,2,3 : transverse momentum of the three leptons
        -phi_1,2,3 : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[px,py,pz,energy] : 4-momentum of each lepton. Since there are three,
        there are three lists
    """

    # Create vector arrays for each particle
    particle1 = vector.arr({"pt": pt_1, "phi": phi_1, "eta": eta_1, "mass": mass_1})
    particle2 = vector.arr({"pt": pt_2, "phi": phi_2, "eta": eta_2, "mass": mass_2})
    particle3 = vector.arr({"pt": pt_3, "phi": phi_3, "eta": eta_3, "mass": mass_3})

    # Extract 4-momentum components for each particle
    particles = [particle1, particle2, particle3]
    components = ['px', 'py', 'pz', 'energy']
    p4_list = []

    for particle in particles:
        for component in components:
            p4_list.append(getattr(particle, component))
    # print(p4_list)
    # assert(5==3)
    
    p4_list_flat = np.array(p4_list).flatten().tolist()
    return p4_list
    # return ([px_1, py_1, pz_1, energy_1, px_2, py_2, pz_2, energy_2, px_3, py_3, pz_3, energy_3])

def motherpair_vals(
    pt_1, pt_2, pt_3,
    phi_1, phi_2, phi_3,
    eta_1, eta_2, eta_3,
    mass_1, mass_2, mass_3
):
    """
    Arguments:
        -pt_1,2,3 : transverse momentum of the three leptons
        -phi_1,2,3 : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output:
        -add_feat_array : mass, pt, eta, phi, px, py, pz, energy of the mother particles
    """

    # Create vector arrays for each particle
    particle1 = vector.arr({"pt": pt_1, "phi": phi_1, "eta": eta_1, "mass": mass_1})
    particle2 = vector.arr({"pt": pt_2, "phi": phi_2, "eta": eta_2, "mass": mass_2})
    particle3 = vector.arr({"pt": pt_3, "phi": phi_3, "eta": eta_3, "mass": mass_3})

    # Create mother particles by summing up the particle vectors
    p4_mother12 = particle1 + particle2
    p4_mother23 = particle2 + particle3
    p4_mother13 = particle1 + particle3

    pairs = ['12', '13', '23']
    motherpairs = [p4_mother12, p4_mother13, p4_mother23]
    features_toadd = ['mass', 'pt', 'eta', 'phi', 'px', 'py', 'pz', 'energy']

    # Initialize the feature list
    features_list = []

    # Populate the feature list
    for feature in features_toadd:
        for i, pair in enumerate(pairs):
            features_list.append(getattr(motherpairs[i], feature))

    # features_list_flat = np.array(features_list).flatten().tolist()
    # print(features_list)
    return features_list

def Energychecker(input, output):
    input_np=input
    output=output
    inputvars2=['1_eta', '1_mass', '1_phi', '1_pt', '2_eta', '2_mass', '2_phi', '2_pt', '3_eta', '3_mass', '3_phi', '3_pt', 'MET_phi', 'MET_pt']
    inputdict={var: val for var, val in zip(inputvars2, input_np)}

    particle1 = vector.obj(phi=inputdict['1_phi'], pt=inputdict['1_pt'], eta=inputdict['1_eta'], mass=inputdict['1_mass'])
    particle2 = vector.obj(phi=inputdict['2_phi'], pt=inputdict['2_pt'], eta=inputdict['2_eta'], mass=inputdict['2_mass'])
    particle3 = vector.obj(phi=inputdict['3_phi'], pt=inputdict['3_pt'], eta=inputdict['3_eta'], mass=inputdict['3_mass'])

    Etot=particle1.E+particle2.E+particle3.E+inputdict['MET_pt']
    print("etot calculated vs predicted", Etot, output)



def Energy_tot(
    pt_1, pt_2, pt_3,
    phi_1, phi_2, phi_3,
    eta_1, eta_2, eta_3,
    mass_1, mass_2, mass_3,
    MET_pt
    
):
    """
    Arguments:
        -pt_1,2,3 : transverse momentum of the three leptons
        -phi_1,2,3 : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output:
        -Energy_tot : total energy of the three leptons
    """

    # Create vector arrays for each particle
    # particle1 = vector.arr({"pt": pt_1, "phi": phi_1, "eta": eta_1, "mass": mass_1})
    # particle2 = vector.arr({"pt": pt_2, "phi": phi_2, "eta": eta_2, "mass": mass_2})
    # particle3 = vector.arr({"pt": pt_3, "phi": phi_3, "eta": eta_3, "mass": mass_3})
    particle1=vector.array({"pt": pt_1, "phi": phi_1, "eta": eta_1, "mass": mass_1})
    particle2=vector.array({"pt": pt_2, "phi": phi_2, "eta": eta_2, "mass": mass_2})
    particle3=vector.array({"pt": pt_3, "phi": phi_3, "eta": eta_3, "mass": mass_3})

    # Create mother particles by summing up the particle vectors
    E_1=particle1.energy
    E_2=particle2.energy
    E_3=particle3.energy

    # Calculate the total energy
    Energy_tot = E_1+E_2+E_3+MET_pt

    return Energy_tot



# print(generate_random_sample(5))
