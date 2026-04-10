import uproot
import numpy as np
import pandas as pd

def dist(file, PID = False, bins = 50, use_abs = False, name = False):
    file = uproot.open(file)
    tree = file["Delphes"]

    pid = tree["Particle/Particle.PID"].array(library = "np")
    pt = tree["Particle/Particle.PT"].array(library = "np")
    eta = tree["Particle/Particle.Eta"].array(library = "np")
    phi = tree["Particle/Particle.Phi"].array(library = "np")
    charge = tree["Particle/Particle.Charge"].array(library = "np")

    events = len(pt)

    if use_abs == True:
        ind = [[i, l] for i in range(events) for l in np.where(abs(pid[i]) == PID)[0]]
        pt_p = [pt[ind[i][0]][ind[i][1]] for i in range(len(ind))]
        eta_p = [eta[ind[i][0]][ind[i][1]] for i in range(len(ind))]
        phi_p = [phi[ind[i][0]][ind[i][1]] for i in range(len(ind))]
        charge_p = [charge[ind[i][0]][ind[i][1]] for i in range(len(ind))]
    else:
        ind = [[i, l] for i in range(events) for l in np.where(pid[i] == PID)[0]]
        pt_p = [pt[ind[i][0]][ind[i][1]] for i in range(len(ind))]
        eta_p = [eta[ind[i][0]][ind[i][1]] for i in range(len(ind))]
        phi_p = [phi[ind[i][0]][ind[i][1]] for i in range(len(ind))]
        charge_p = [charge[ind[i][0]][ind[i][1]] for i in range(len(ind))]

    if name:
        data = {
            "P_t" + " "*1 + name: pt_p,
            "Eta" + " "*1 + name: eta_p,
            "Phi" + " "*1 + name: phi_p,
            "Charge" + " "*1 + name: charge_p
        }
    else:
        data = {
            "P_t": pt_p,
            "Eta": eta_p,
            "Phi": phi_p,
            "Charge": charge_p
        }
    data = pd.DataFrame(data)   
    data.hist(bins = bins) 
    return data