#!/usr/bin/env python3
"""Portierung von Auswertung.ogs (metrics.ogs + mitteln.ogs) nach Python.

Nur die Kennzahlen-Logik -- Datei-Ein-/Ausgabe (Metadaten aus dem
CSV-Header-JSON) kommt als eigener Schritt dazu.
"""
import statistics as st

# --- Parameter, identisch zu Auswertung.ogs ---------------------------------
THRES  = 0.95   # Bruchteil von M*, der als "erreicht" gilt
MINACC = 25.0   # M* darunter = Lauf hat nie gelernt
NWIN   = 10     # Fensterbreite W fuer alle gleitenden Mediane
NHOLD  = 3      # so viele Epochen in Folge muessen halten
DEGREL = 0.10   # relativer Verlust, ab dem "degradiert" gilt


def metrics(acc):
    '''Die vier Kennzahlen einer einzelnen Kurve (ein Seed oder die Mittelkurve).

    M*       bestes gleitendes Fenster: max ueber alle Fensterpositionen des
             Fenstermedians. Nicht max(acc) -- ein einzelner Ausreisser nach
             oben soll den Bestwert nicht setzen.
    M_end    Median des Endfensters, also wo die Kurve aufhoert.
    learned  M* >= MINACC. Ein Lauf auf Zufallsniveau (~10 %) faellt durch.
    degraded gelernt, aber am Ende mehr als DEGREL unter dem Bestwert.
    '''
    n = len(acc)
    if n < NWIN:
        return -1.0, -1.0, False, False

    mstar = max(st.median(acc[i:i + NWIN]) for i in range(n - NWIN + 1))
    mend = st.median(acc[n - NWIN:])
    learned = mstar >= MINACC
    degraded = learned and (mstar - mend) > DEGREL * mstar
    return mstar, mend, learned, degraded


def e_conv(acc, mstar):
    '''E_conv: erste Epoche (1-basiert), ab der die Kurve haelt.

    Vorwaertsscan wie in Auswertung.ogs -- nicht nur "erstes Fenster ueber
    der Schwelle", sondern mitgefuehrt, ob die Kurve VOR der ersten
    Halte-Serie schon einmal darunter lag:
      -1  nie gelernt (M* < MINACC), oder die nHold-Serie wird nie erreicht
       0  Serie erreicht, aber die Kurve war nie vorher darunter -- von
          Anfang an auf Niveau, keine echte Lernphase
      >0  echte Konvergenzepoche (die Kurve war vorher mal darunter)
    '''
    if mstar < MINACC:
        return -1
    target = THRES * mstar
    run_len = 0
    was_below = False
    for i, val in enumerate(acc, start=1):
        if val >= target:
            run_len += 1
            if run_len >= NHOLD:
                return (i - NHOLD + 1) if was_below else 0
        else:
            run_len = 0
            was_below = True
    return -1


def ripple(acc, start_epoch):
    '''S_ripple: robuste Std-Schaetzung (MAD/0.6745) im Fenster [start_epoch..N].

    start_epoch ist 1-basiert (wie E_conv); 0 bedeutet "ganze Kurve ist
    Plateau" und wird wie 1 behandelt. Median/MAD statt Mittelwert/Std,
    damit ein einzelner Ausreisser das Rauschmass nicht verzerrt.
    '''
    window = acc[max(start_epoch, 1) - 1:]
    med = st.median(window)
    mad = st.median([abs(x - med) for x in window])
    return mad / 0.6745


class Seed:
    def __init__(self, a):
        self.a = a
        self.mstar, self.mend, self.learned, self.degraded = metrics(self.a)
        self.e_conv = e_conv(self.a, self.mstar)


class Run:
    def __init__(self, a1, a2, a3):
        self.s1 = Seed(a1)
        self.s2 = Seed(a2)
        self.s3 = Seed(a3)
        self.calc_metrics()

    def calc_metrics(self):
        '''Study-Kennzahlen aus den drei Seeds, aequivalent zu Auswertung.ogs.

        Mstar_mean/min/max/span laufen ueber ALLE drei rohen M*-Werte -- ein
        nicht lernender Seed zieht den Mittelwert ehrlich runter.
        Mstar_meancurve/E_conv/S_ripple laufen dagegen nur ueber die LERNER
        (Seeds mit learned=True); bei nLearned=0 bleiben sie auf -1.
        nLost zaehlt gelernte UND degradierte Seeds, nicht "nicht gelernt".
        S_ripple nimmt je Lerner das Fenster [E_conv..N] der STUDY (nicht des
        einzelnen Seeds) und dann das Maximum ueber die Lerner.
        '''
        seeds = (self.s1, self.s2, self.s3)
        self.n_learned = sum(s.learned for s in seeds)
        self.n_lost = sum(s.degraded for s in seeds)
        self.seed_dependent = 0 < self.n_learned < 3

        seed_mstars = [s.mstar for s in seeds]
        self.mstar_mean = sum(seed_mstars) / 3
        self.mstar_min = min(seed_mstars)
        self.mstar_max = max(seed_mstars)
        self.mstar_span = self.mstar_max - self.mstar_min

        learners = [s for s in seeds if s.learned]
        if learners:
            self.mean_curve = [sum(vals) / len(learners) for vals in zip(*(s.a for s in learners))]
            self.mstar_meancurve, _, _, _ = metrics(self.mean_curve)
            self.e_conv = e_conv(self.mean_curve, self.mstar_meancurve)
        else:
            self.mean_curve = None
            self.mstar_meancurve = -1.0
            self.e_conv = -1

        self.e_conv_n_seed = sum(1 for s in seeds if s.e_conv > 0.5)

        if self.n_learned > 0 and self.e_conv != -1:
            self.ripple = max(ripple(s.a, self.e_conv) for s in learners)
        else:
            self.ripple = -1.0
