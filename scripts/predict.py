"""CLI: python scripts/predict.py --airline AA --from JFK --to LAX --day 5 --time 1140 --length 240"""

import argparse

from flight_delay import predict


def main():
    parser = argparse.ArgumentParser(description="Uçuş gecikme tahmini")
    parser.add_argument("--airline", required=True, help="Havayolu kodu, örn. AA")
    parser.add_argument("--from", dest="airport_from", required=True, help="Kalkış havalimanı kodu")
    parser.add_argument("--to", dest="airport_to", required=True, help="Varış havalimanı kodu")
    parser.add_argument("--day", type=int, required=True, help="Haftanın günü (1-7)")
    parser.add_argument("--time", type=int, required=True, help="Planlanan kalkış (dakika, gece yarısından itibaren)")
    parser.add_argument("--length", type=int, required=True, help="Planlanan uçuş süresi (dakika)")
    args = parser.parse_args()

    sample = {
        "Airline": args.airline,
        "AirportFrom": args.airport_from,
        "AirportTo": args.airport_to,
        "DayOfWeek": args.day,
        "Time": args.time,
        "Length": args.length,
    }

    proba, label = predict.predict_new(sample)
    print(f"Gecikme olasılığı: {proba:.2%}")
    print(f"Sonuç: {label}")


if __name__ == "__main__":
    main()
