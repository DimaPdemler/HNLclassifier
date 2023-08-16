from numpy import array, zeros_like, zeros, sign, sqrt, cos, sin, arctan, exp, pi
import numpy as np
import random
from vector import array as v_array
from vector import arr



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
    charge_1,
    charge_2,
    charge_3,
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
        -charge_1,2,3 : charge of the three leptons
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[angle1, angle2] : angles between 2 leptons of opposite sign (candidate for HNL desintegration) in their rest frame.
                            Since the total charge of the 3 leptons is +/-1, there's 2 choice of lepton pair -> 2 angles in the output
    """


    n = len(charge_1)

    eta_MET = []
    mass_MET = []
    if type(charge_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(charge_1)
        mass_MET = zeros_like(charge_1)

    vector_MET = arr({"pt": pt_MET, "phi": phi_MET, "eta": eta_MET, "M": mass_MET})

    indices = [1, 2, 3]

    pts = dict(zip(indices, [[], [], []]))
    phis = dict(zip(indices, [[], [], []]))
    etas = dict(zip(indices, [[], [], []]))
    masses = dict(zip(indices, [[], [], []]))

    for i in range(len(charge_1)):
        charge_tot = charge_1[i] + charge_2[i] + charge_3[i]

        if sign(charge_tot) != charge_1[i]:
            pts[1].append(pt_1[i])
            pts[2].append(pt_2[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_1[i])
            phis[2].append(phi_2[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_1[i])
            etas[2].append(eta_2[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_1[i])
            masses[2].append(mass_2[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_2[i]:
            pts[1].append(pt_2[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_2[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_2[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_2[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_3[i]:
            pts[1].append(pt_3[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_2[i])
            phis[1].append(phi_3[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_2[i])
            etas[1].append(eta_3[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_2[i])
            masses[1].append(mass_3[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_2[i])

    angles = []
    pair_candidate = [[1, 2], [1, 3]]

    for pair in pair_candidate:
        i = pair[0]
        j = pair[1]
        vector_i = arr({"pt": pts[i], "phi": phis[i], "eta": etas[i], "M": masses[i]})
        vector_j = arr({"pt": pts[j], "phi": phis[j], "eta": etas[j], "M": masses[j]})

        vector_tot = vector_i + vector_j + vector_MET
        vector_i = vector_i.boostCM_of_p4(vector_tot)
        vector_j = vector_j.boostCM_of_p4(vector_tot)
        # print(vector_i.phi.shape, vector_j.phi.shape)

        angle = vector_i.deltaangle(vector_j)
        angles.append(angle)
    return angles


def W_CM_angles_to_plane(
    charge_1,
    charge_2,
    charge_3,
    pt_1,
    pt_2,
    pt_3,
    phi_1,
    phi_2,
    phi_3,
    eta_1,
    eta_2,
    eta_3,
    mass_1,
    mass_2,
    mass_3,
):
    """
    Arguments :
        -charge_1,2,3 : charge of the three leptons
        -pt_1,2,3 : transverse momentum of the three leptons
        -phi_1,2,3 : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[angle1, angle2] : angles between 1 lepton and the plane formed by 2 other leptons of opposite sign (candidate for HNL
                            desintegration) in the rest frame of the 3 leptons. Since the total charge of the 3 leptons is +/-1,
                            there's 2 choice of lepton pair -> 2 angles in the output
    """

    indices = [1, 2, 3]

    pts = dict(zip(indices, [[], [], []]))
    phis = dict(zip(indices, [[], [], []]))
    etas = dict(zip(indices, [[], [], []]))
    masses = dict(zip(indices, [[], [], []]))

    for i in range(len(charge_1)):
        charge_tot = charge_1[i] + charge_2[i] + charge_3[i]
        if sign(charge_tot) != charge_1[i]:
            pts[1].append(pt_1[i])
            pts[2].append(pt_2[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_1[i])
            phis[2].append(phi_2[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_1[i])
            etas[2].append(eta_2[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_1[i])
            masses[2].append(mass_2[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_2[i]:
            pts[1].append(pt_2[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_2[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_2[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_2[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_3[i]:
            pts[1].append(pt_3[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_2[i])
            phis[1].append(phi_3[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_2[i])
            etas[1].append(eta_3[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_2[i])
            masses[1].append(mass_3[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_2[i])

    angles = []
    pair_candidate = [[1, 2], [1, 3]]

    for pair in pair_candidate:
        i = pair[0]
        j = pair[1]
        k = next(value for value in [1, 2, 3] if value != i and value != j)

        vector_first = arr(
            {"pt": pts[k], "phi": phis[k], "eta": etas[k], "M": masses[k]}
        )
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
    charge_1,
    charge_2,
    charge_3,
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
        -charge_1,2,3 : charge of the three leptons
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[angle1, angle2] : angles between 1 lepton and the plane formed by 2 other leptons of opposite sign (candidate for HNL
                            desintegration) in the rest frame of the 3 leptons. Since the total charge of the 3 leptons is +/-1,
                            there's 2 choice of lepton pair -> 2 angles in the output
    """

    n = len(charge_1)
    eta_MET = []
    mass_MET = []
    if type(charge_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(charge_1)
        mass_MET = zeros_like(charge_1)
    vector_MET = arr({"pt": pt_MET, "phi": phi_MET, "eta": eta_MET, "M": mass_MET})

    indices = [1, 2, 3]

    pts = dict(zip(indices, [[], [], []]))
    phis = dict(zip(indices, [[], [], []]))
    etas = dict(zip(indices, [[], [], []]))
    masses = dict(zip(indices, [[], [], []]))

    for i in range(len(charge_1)):
        charge_tot = charge_1[i] + charge_2[i] + charge_3[i]
        if sign(charge_tot) != charge_1[i]:
            pts[1].append(pt_1[i])
            pts[2].append(pt_2[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_1[i])
            phis[2].append(phi_2[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_1[i])
            etas[2].append(eta_2[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_1[i])
            masses[2].append(mass_2[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_2[i]:
            pts[1].append(pt_2[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_2[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_2[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_2[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_3[i]:
            pts[1].append(pt_3[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_2[i])
            phis[1].append(phi_3[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_2[i])
            etas[1].append(eta_3[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_2[i])
            masses[1].append(mass_3[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_2[i])

    angles = []
    pair_candidate = [[1, 2], [1, 3]]

    for pair in pair_candidate:
        i = pair[0]
        j = pair[1]
        k = next(value for value in [1, 2, 3] if value != i and value != j)

        vector_first = arr(
            {"pt": pts[k], "phi": phis[k], "eta": etas[k], "M": masses[k]}
        )
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
    charge_1,
    charge_2,
    charge_3,
    pt_1,
    pt_2,
    pt_3,
    phi_1,
    phi_2,
    phi_3,
    eta_1,
    eta_2,
    eta_3,
    mass_1,
    mass_2,
    mass_3,
):
    """
    Arguments :
        -charge_1,2,3 : charge of the three leptons
        -pt_1,2,3 : transverse momentum of the three leptons
        -phi_1,2,3 : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[HNL_mass1, HNL_mass2] : invariant mass of the sum of 2 leptons of opposite sign (candidate for HNL desintegration).
                        Since the total charge of the 3 leptons is +/-1, there's 2 choice of lepton pair -> 2 masses in the output
    """

    indices = [1, 2, 3]

    pts = dict(zip(indices, [[], [], []]))
    phis = dict(zip(indices, [[], [], []]))
    etas = dict(zip(indices, [[], [], []]))
    masses = dict(zip(indices, [[], [], []]))

    for i in range(len(charge_1)):
        charge_tot = charge_1[i] + charge_2[i] + charge_3[i]
        if sign(charge_tot) != charge_1[i]:
            pts[1].append(pt_1[i])
            pts[2].append(pt_2[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_1[i])
            phis[2].append(phi_2[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_1[i])
            etas[2].append(eta_2[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_1[i])
            masses[2].append(mass_2[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_2[i]:
            pts[1].append(pt_2[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_2[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_2[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_2[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_3[i]:
            pts[1].append(pt_3[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_2[i])
            phis[1].append(phi_3[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_2[i])
            etas[1].append(eta_3[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_2[i])
            masses[1].append(mass_3[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_2[i])

    HNL_masses = []
    pair_candidate = [[1, 2], [1, 3]]

    for pair in pair_candidate:
        i = pair[0]
        j = pair[1]

        vector_i = arr({"pt": pts[i], "phi": phis[i], "eta": etas[i], "M": masses[i]})
        vector_j = arr({"pt": pts[j], "phi": phis[j], "eta": etas[j], "M": masses[j]})
        vector_tot = vector_i + vector_j
        HNL_masses.append(vector_tot.mass)
    return HNL_masses


def HNL_CM_masses_with_MET(
    charge_1,
    charge_2,
    charge_3,
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
        -charge_1,2,3 : charge of the three leptons
        -pt_1,2,3,MET : transverse momentum of the three leptons and the missing momentum
        -phi_1,2,3,MET : azimuthal angle of the three leptons
        -eta_1,2,3 : pseudorapidity of the three leptons
        -mass_1,2,3 : mass of the three leptons
    Output :
        -[HNL_mass1, HNL_mass2] : invariant mass of the sum of 2 leptons of opposite sign (candidate for HNL desintegration).
                        Since the total charge of the 3 leptons is +/-1, there's 2 choice of lepton pair -> 2 masses in the output
    """

    n = len(charge_1)
    eta_MET = []
    mass_MET = []
    if type(charge_1) == list:
        for i in range(n):
            eta_MET.append(0)
            mass_MET.append(0)
    else:
        eta_MET = zeros_like(charge_1)
        mass_MET = zeros_like(charge_1)
    vector_MET = arr({"pt": pt_MET, "phi": phi_MET, "eta": eta_MET, "M": mass_MET})

    indices = [1, 2, 3]

    pts = dict(zip(indices, [[], [], []]))
    phis = dict(zip(indices, [[], [], []]))
    etas = dict(zip(indices, [[], [], []]))
    masses = dict(zip(indices, [[], [], []]))

    for i in range(len(charge_1)):
        charge_tot = charge_1[i] + charge_2[i] + charge_3[i]
        if sign(charge_tot) != charge_1[i]:
            pts[1].append(pt_1[i])
            pts[2].append(pt_2[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_1[i])
            phis[2].append(phi_2[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_1[i])
            etas[2].append(eta_2[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_1[i])
            masses[2].append(mass_2[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_2[i]:
            pts[1].append(pt_2[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_3[i])
            phis[1].append(phi_2[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_3[i])
            etas[1].append(eta_2[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_3[i])
            masses[1].append(mass_2[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_3[i])
        if sign(charge_tot) != charge_3[i]:
            pts[1].append(pt_3[i])
            pts[2].append(pt_1[i])
            pts[3].append(pt_2[i])
            phis[1].append(phi_3[i])
            phis[2].append(phi_1[i])
            phis[3].append(phi_2[i])
            etas[1].append(eta_3[i])
            etas[2].append(eta_1[i])
            etas[3].append(eta_2[i])
            masses[1].append(mass_3[i])
            masses[2].append(mass_1[i])
            masses[3].append(mass_2[i])

    HNL_masses = []
    pair_candidate = [[1, 2], [1, 3]]

    for pair in pair_candidate:
        i = pair[0]
        j = pair[1]

        vector_i = arr({"pt": pts[i], "phi": phis[i], "eta": etas[i], "M": masses[i]})
        vector_j = arr({"pt": pts[j], "phi": phis[j], "eta": etas[j], "M": masses[j]})
        vector_tot = vector_i + vector_j + vector_MET
        HNL_masses.append(vector_tot.mass)
    return HNL_masses

def generate_random_sample(num_events=100):
    pt_muon = np.random.uniform(0, 10, num_events) 
    pt_tau = np.random.uniform(0, 10, num_events) 
    pt_MET = np.random.uniform(0, 10, num_events) 
    eta_muon = np.random.uniform(-5, 5, num_events) 
    eta_tau = np.random.uniform(-5, 5, num_events) 
    eta_MET = np.random.uniform(-5, 5, num_events) 
    phi_muon = np.random.uniform(-np.pi, np.pi, num_events) 
    phi_tau = np.random.uniform(-np.pi, np.pi, num_events) 
    phi_MET = np.random.uniform(-np.pi, np.pi, num_events) 
    m_muon = np.random.uniform(0, 1, num_events) 
    m_tau = np.random.uniform(0, 1, num_events) 
    m_MET = np.random.uniform(0, 1, num_events) 

     # Group the parameters by type
    pts = [pt_muon, pt_tau, pt_MET]
    phis = [phi_muon, phi_tau, phi_MET]
    etas = [eta_muon, eta_tau, eta_MET]
    masses = [m_muon, m_tau, m_MET]

    # Calculate the quantities
    inv_mass = invariant_mass(pts, phis, etas, masses)


    # Return the quantities as a dictionary
    return {
        'invariantMass': inv_mass
    }


# print(generate_random_sample(5))
