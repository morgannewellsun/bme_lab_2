
import argparse
from pprint import pprint as python_pprint

from neuron import h
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import linear_model

plt.rcParams["figure.figsize"] = [6, 6]
plt.rcParams["figure.autolayout"] = True


def pprint(name, value, decimals=4):
    print(f"{name:8} = {value:8.{decimals}f}")


def length_constant(radius, membrane_resistance, axial_resistance):
    return np.sqrt((radius * membrane_resistance) / (2 * axial_resistance))


class AllCompartmentsRecorder:

    def __init__(self, section, variable_name):

        self.nseg = section.nseg
        self.reference_name = "_ref_" + variable_name
        self.compartment_vectors = []

        for idx in range(section.nseg):
            self.compartment_vectors.append(
                h.Vector().record(
                    getattr(
                        section(self.idx_to_frac(idx)),
                        self.reference_name)))

    def __getitem__(self, idx):
        return self.compartment_vectors[idx]

    def idx_to_frac(self, x):
        return (x + 0.5) / self.nseg

    def frac_to_idx(self, x):
        return np.clip(x * self.nseg, 0, self.nseg - 1).astype(int)


def main(output_dir):

    # Calculate length constants for two axons
    print("\nLength Constants: ")
    pprint("radius  5um", length_constant(radius=5, membrane_resistance=40000, axial_resistance=200))
    pprint("radius 10um", length_constant(radius=10, membrane_resistance=40000, axial_resistance=200))

    # Specify and simulate both axons
    results_dict = {}
    for diam in (10, 20):

        axon = h.Section(name="axon")
        axon.L = 5000
        axon.nseg = 1001
        axon.insert("hh").insert("pas")
        axon.Ra = 200
        axon.diam = diam

        iclamp = h.IClamp(axon(0.0))
        iclamp.delay = 7
        iclamp.dur = 0.2
        iclamp.amp = 30
        recorder_voltage_all = AllCompartmentsRecorder(axon, "v")
        recorder_time = h.Vector().record(h._ref_t)
        h.load_file("stdrun.hoc")
        h.finitialize(-65)
        h.continuerun(20)

        voltage_compartment_time_array = np.array(recorder_voltage_all.compartment_vectors)
        ap_peak_indices = np.argmax(voltage_compartment_time_array, axis=0)
        ap_peak_positions = recorder_voltage_all.idx_to_frac(ap_peak_indices) * axon.L
        ap_exists = np.max(voltage_compartment_time_array, axis=0) > 20
        results_dict[diam] = {
            "recorder_voltage_all": recorder_voltage_all,
            "recorder_time": recorder_time,
            "ap_peak_positions": ap_peak_positions,
            "ap_exists": ap_exists}

        ransac = linear_model.RANSACRegressor()
        ransac.fit(
            X=np.array(recorder_time)[ap_exists][:, np.newaxis],
            y=ap_peak_positions[ap_exists])
        print(f"\nAP propagation speed for radius {int(diam/2)} (um/s):")
        print(ransac.estimator_.coef_[0])

    # Plot action potentials at various points along both axons
    fig, axes = plt.subplots(2, 1, sharex="all", sharey="all")
    axes[0].set_title(f"ap voltage over time")
    axes[-1].set_xlabel(r"time $(\mu s)$")
    axes[-1].set_ylim(-80, 50)
    for pos in np.linspace(start=0.0, stop=1.0, num=9):
        for idx, diam in enumerate((10, 20)):
            axes[idx].set_ylabel(r"$V_{m}$  $(mV)$")
            axes[idx].plot(
                results_dict[diam]["recorder_time"],
                results_dict[diam]["recorder_voltage_all"][results_dict[diam]["recorder_voltage_all"].frac_to_idx(pos)],
                label=f"pos={pos:1.1f}, a={int(diam/2)}",
                color=cm.get_cmap("viridis")(1 - pos),
                alpha=0.8)
            axes[idx].legend(loc="upper left")
    plt.savefig(output_dir + f"comparison_ap.png")
    plt.close()

    # Plot action potential position over time for both axons
    fig, axes = plt.subplots(1, 1)
    axes.set_title(f"ap position over time")
    axes.set_xlabel(r"time $(\mu s)$")
    axes.set_ylabel(r"AP pos ($\mu m$)")
    axes.plot(
        np.array(results_dict[10]["recorder_time"])[results_dict[10]["ap_exists"]],
        results_dict[10]["ap_peak_positions"][results_dict[10]["ap_exists"]],
        label=r"$a=5 \mu m$")
    axes.plot(
        np.array(results_dict[20]["recorder_time"])[results_dict[20]["ap_exists"]],
        results_dict[20]["ap_peak_positions"][results_dict[20]["ap_exists"]],
        label=r"$a=10 \mu m$")
    axes.legend()
    plt.savefig(output_dir + f"comparison_speed.png")
    plt.close()

    del axon
    results_dict = {}
    for run_params in ((1, 1), (1, 10), (0.5, 10), (2, 10)):

        geometry_scale, channel_scale = run_params

        # Specify a neuron
        dendrite = h.Section(name="dendrite")
        dendrite.nseg = 222
        dendrite.L = 50 * geometry_scale
        dendrite.diam = 12 * geometry_scale
        soma = h.Section(name="soma")
        soma.nseg = 100
        soma.L = 24 * geometry_scale
        soma.diam = 21 * geometry_scale
        soma.connect(dendrite)
        axon_hillock = h.Section(name="axon_hillock")
        axon_hillock.nseg = 9
        axon_hillock.L = 16 * geometry_scale
        axon_hillock_diameters = np.linspace(start=21, stop=1, num=9)
        for i, segment in enumerate(axon_hillock):
            segment.diam = axon_hillock_diameters[i] * geometry_scale
        axon_hillock.connect(soma)
        axon_nonmyelinated = h.Section(name="axon_nonmyelinated")
        axon_nonmyelinated.nseg = 100
        axon_nonmyelinated.L = 16 * geometry_scale
        axon_nonmyelinated.diam = 1 * geometry_scale
        axon_nonmyelinated.connect(axon_hillock)
        axon_myelinated = h.Section(name="axon_myelinated")
        axon_myelinated.nseg = 100
        axon_myelinated.L = 300 * geometry_scale
        axon_myelinated.diam = 1 * geometry_scale
        axon_myelinated.connect(axon_nonmyelinated)
        all_sections = [dendrite, soma, axon_hillock, axon_nonmyelinated, axon_myelinated]
        for section in all_sections:
            section.insert("pas")
            section.insert("extracellular")
            section.insert("hh") if section is not axon_myelinated else None
        axon_myelinated.cm = 0.04
        for segment in axon_nonmyelinated:
            segment.hh.gnabar *= channel_scale
            segment.hh.gkbar *= channel_scale

        # Identify threshold stimulus current using binary search
        iclamp = h.IClamp(dendrite(0.0))
        iclamp.delay = 2
        iclamp.dur = 0.1
        iters = 32
        stimulus_bound_upper = 40
        stimulus_bound_lower = 0
        for _ in range(iters):
            stimulus_magnitude = (stimulus_bound_upper + stimulus_bound_lower) / 2
            iclamp.amp = stimulus_magnitude
            m_recorder = h.Vector().record(axon_hillock(0.5).hh._ref_m)
            h.load_file("stdrun.hoc")
            h.finitialize(-65)
            h.continuerun(10)
            ap_exists = np.max(np.array(m_recorder)) > 0.9
            if ap_exists:
                stimulus_bound_upper = stimulus_magnitude
            else:
                stimulus_bound_lower = stimulus_magnitude
        print(f"\nThreshold Stim Current (mA/cm2) for {geometry_scale:1.1f}x geometry, {channel_scale:2.0f}x non-myel. axon channels:")
        pprint("UBound", stimulus_bound_upper, decimals=8)
        pprint("LBound", stimulus_bound_lower, decimals=8)

        # Run a final simulation just above the threshold stimulus
        stimulus_magnitude = stimulus_bound_upper * 1.1
        iclamp.amp = stimulus_magnitude
        data = {
            "time": h.Vector().record(h._ref_t),
            "dendrite": h.Vector().record(dendrite(0.5)._ref_i_membrane),
            "soma": h.Vector().record(soma(0.5)._ref_i_membrane),
            "axon_hillock": h.Vector().record(axon_hillock(0.5)._ref_i_membrane),
            "axon_nonmyelinated": h.Vector().record(axon_nonmyelinated(0.5)._ref_i_membrane),
            "axon_myelinated": h.Vector().record(axon_myelinated(0.5)._ref_i_membrane)}
        h.load_file("stdrun.hoc")
        h.finitialize(-65)
        h.continuerun(10)
        data.update({
            "dendrite_abs": data["dendrite"] * dendrite(0.5).area() * 1e-8,
            "soma_abs": data["soma"] * soma(0.5).area() * 1e-8,
            "axon_hillock_abs": data["axon_hillock"] * axon_hillock(0.5).area() * 1e-8,
            "axon_nonmyelinated_abs": data["axon_nonmyelinated"] * axon_nonmyelinated(0.5).area() * 1e-8,
            "axon_myelinated_abs": data["axon_myelinated"] * axon_myelinated(0.5).area() * 1e-8,
        })
        results_dict[run_params] = data

    # Plot comparison of current magnitudes at different points in the cell
    for idx, channel_scale in enumerate((10, 1)):
        fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
        axes.set_title(f"per-area current at {channel_scale}x non-myel. axon channel density")
        axes.set_xlabel(r"time $(\mu s)$")
        axes.set_ylabel(r"$I_m$ ($mA/cm^2$)")
        for section_name in ("axon_nonmyelinated", "dendrite", "axon_myelinated", "soma", "axon_hillock"):
            axes.plot(
                results_dict[(1, channel_scale)]["time"],
                results_dict[(1, channel_scale)][section_name],
                label=f"{section_name}")
        axes.legend()
        plt.savefig(output_dir + f"comparison_channels_{channel_scale}.png")
        plt.close()

    # Plot comparison of currents with and without the 10x channel modifier
    for section_name in ("dendrite", "soma", "axon_hillock", "axon_nonmyelinated", "axon_myelinated"):
        fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
        axes.set_title(f"per-area {section_name} current")
        axes.set_xlabel(r"time $(\mu s)$")
        axes.set_ylabel(r"$I_m$ ($mA/cm^2$)")
        for idx, channel_scale in enumerate((10, 1)):
            axes.plot(
                results_dict[(1, channel_scale)]["time"],
                results_dict[(1, channel_scale)][section_name],
                label=f"{channel_scale}x non-myel. axon channels")
        axes.legend()
        plt.savefig(output_dir + f"comparison_channels_{section_name}.png")
        plt.close()

    # Plot comparison of currents at different geometry scales
    for section_name in ("dendrite", "soma", "axon_hillock", "axon_nonmyelinated", "axon_myelinated"):
        fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
        axes.set_title(f"per-area {section_name} current")
        axes.set_xlabel(r"time $(\mu s)$")
        axes.set_ylabel(r"$I_m$ ($mA/cm^2$)")
        for idx, geometry_scale in enumerate((2, 1, 0.5)):
            axes.plot(
                results_dict[(geometry_scale, 10)]["time"],
                results_dict[(geometry_scale, 10)][section_name],
                label=f"{geometry_scale}x geometry")
        axes.legend()
        plt.savefig(output_dir + f"comparison_geometries_{section_name}.png")
        plt.close()

    # Plot comparison of currents at different geometry scales, accounting for surface area
    for section_name in ("dendrite", "soma", "axon_hillock", "axon_nonmyelinated", "axon_myelinated"):
        fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
        axes.set_title(f"absolute {section_name} current")
        axes.set_xlabel(r"time $(\mu s)$")
        axes.set_ylabel(r"$I_m$ ($mA$)")
        for idx, geometry_scale in enumerate((2, 1, 0.5)):
            axes.plot(
                results_dict[(geometry_scale, 10)]["time"],
                results_dict[(geometry_scale, 10)][section_name + "_abs"],
                label=f"{geometry_scale}x geometry")
        axes.legend()
        plt.savefig(output_dir + f"comparison_geometries_abs_{section_name}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="where to output plot images")
    args = parser.parse_args()
    main(args.output_dir)