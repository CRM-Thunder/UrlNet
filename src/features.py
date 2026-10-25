
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit
import ipaddress
import pandas as pd
import tldextract


# ========== KONFIGURACJA ==========

INPUT_FILE = Path("./../data/deduplicated_corrected_phishing_dataset.csv")
OUTPUT_FILE = Path("./../data/urls_features.csv")

URL_COLUMN = "url"
extractor = tldextract.TLDExtract(suffix_list_urls=())

TOKEN_SEPARATOR = re.compile(r"[/\-._?=&]+")
SCHEME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")


# ========== FUNKCJE POMOCNICZE ==========

def safe_ratio(numerator, denominator):
    """Zwraca 0, gdy mianownik wynosi zero."""
    return numerator / denominator if denominator != 0 else 0


def shannon_entropy(text):
    """Entropia Shannona znaków; pusty tekst ma entropię 0."""
    if not text:
        return 0

    counts = Counter(text)
    length = len(text)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )

def get_ip_version(hostname):
    """
    Zwraca:
    4 - dla IPv4
    6 - dla IPv6
    0 - jeśli host nie jest adresem IP
    """
    try:
        return ipaddress.ip_address(hostname).version
    except ValueError:
        return 0


def parse_url(url):
    """
    Parsowanie URL-a z uwzględnieniem braku schematu.

    Dodanie // jest wyłącznie pomocnicze i nie zmienia
    oryginalnego URL-a ani jego długości.
    """
    if SCHEME_PATTERN.match(url) or url.startswith("//"):
        return urlsplit(url)

    try:
        return urlsplit("//" + url)
    except ValueError:
        print(url)
        exit(1)


# ========== EKSTRAKCJA CECH ==========

def extract_features(url):
    """Wyodrębnia cechy z pojedynczego URL-a."""

    if not isinstance(url, str):
        url = ""

    original_url = url

    # Zachowujemy oryginalny zapis przy obliczaniu
    # długości, liczników znaków i HTTPS.
    parsed = parse_url(original_url)

    hostname = parsed.hostname or ""
    path = parsed.path
    query = parsed.query

    # Domeny i subdomeny
    # Rozpoznanie typu hosta
    ip_version = get_ip_version(hostname)
    is_ip = int(ip_version != 0)

    # Długość całego hosta, niezależnie od jego typu
    host_length = len(hostname)

    if is_ip:
        # Adres IP nie ma domeny, TLD ani subdomeny
        registered_domain = ""
        tld = ""
        subdomain = ""

    else:
        # Standardowa ekstrakcja domeny
        extracted = extractor(hostname)

        registered_domain = (
            f"{extracted.domain}.{extracted.suffix}"
            if extracted.domain and extracted.suffix
            else extracted.domain or ""
        )

        tld = (
            extracted.suffix.rsplit(".", 1)[-1]
            if extracted.suffix else ""
        )

        subdomain = extracted.subdomain or ""

    # Długości
    url_length = len(original_url)
    domain_length = len(registered_domain)
    arguments_length = len(query)

    # Tokenizacja wyłącznie ścieżki.
    # Usuwamy puste tokeny.
    url_tokens = [
        token
        for token in TOKEN_SEPARATOR.split(original_url)
        if token
    ]

    token_lengths = [len(token) for token in url_tokens]

    longest_token_length = max(token_lengths, default=0)

    average_token_length = (
        sum(token_lengths) / len(token_lengths)
        if token_lengths else 0
    )

    # Ekstrakcja cech
    features = {
        # 1–3. Domeny
        "domain": registered_domain,
        "tld": tld,
        "subdomain": subdomain,

        # 4–6. Długości
        "url_length": url_length,
        "domain_length": domain_length,
        "host_length": host_length,

        # 7–9. Stosunki długości
        "arguments_url_ratio": safe_ratio(
            arguments_length, url_length
        ),
        "arguments_domain_ratio": safe_ratio(
            arguments_length, domain_length
        ),
        "arguments_hostname_ratio": safe_ratio(
            arguments_length, host_length
        ),

        # 10–11. Tokeny ścieżki
        "longest_url_token_length": longest_token_length,
        "average_url_token_length": average_token_length,

        # 12–17. Liczba wybranych znaków w całym URL-u
        "dot_count": original_url.count("."),
        "hyphen_count": original_url.count("-"),
        "underscore_count": original_url.count("_"),
        "question_mark_count": original_url.count("?"),
        "equal_sign_count": original_url.count("="),
        "ampersand_count": original_url.count("&"),

        # 18. Czy URL zaczyna się od HTTPS
        "has_https": int(
            original_url.startswith("https://")
        ),

        # 19. Entropia domeny
        "domain_entropy": shannon_entropy(registered_domain),

        # 20–21. Wszystkie symbole i cyfry w URL-u
        "symbol_count": sum(
            not char.isalnum() for char in original_url
        ),
        "digit_count": sum(
            char.isdigit() for char in original_url
        ),
        #22-23 dotyczy adresów IP
        "is_ip": is_ip,
        "ip_version": ip_version,
    }

    return features


# ========== PRZETWARZANIE DATASETU ==========

def main():
    # 1. Wczytanie danych
    df = pd.read_csv(INPUT_FILE, keep_default_na=False)

    if URL_COLUMN not in df.columns:
        raise KeyError(
            f"Nie znaleziono kolumny '{URL_COLUMN}'. "
            f"Dostępne kolumny: {list(df.columns)}"
        )

    print(f"Wczytano {len(df)} rekordów.")

    # 2. Ekstrakcja
    features_df = pd.DataFrame(
        df[URL_COLUMN].apply(extract_features).tolist(),
        index=df.index
    )

    # 3. Dodanie cech do oryginalnego DataFrame
    df = pd.concat([df, features_df], axis=1)

    # 4. Zapis
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # 5. Podsumowanie
    print(f"Wyekstraktowano {features_df.shape[1]} cech.")
    print(f"Zapisano dane do: {OUTPUT_FILE}")
    print("\nBrakujące wartości w nowych cechach:")
    print(features_df.isna().sum())


if __name__ == "__main__":
    main()
