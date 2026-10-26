
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
label_mapping = {
    'benign': 0,
    'defacement': 1,
    'phishing': 2,
    'malware': 3
}

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

    # Zachowujemy oryginalny zapis przy obliczaniu
    # długości, liczników znaków i HTTPS.
    parsed = parse_url(url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    # Domeny i subdomeny
    # Rozpoznanie typu hosta
    ip_version = get_ip_version(hostname)
    is_ip = int(ip_version != 0)

    # Długość całego hosta, niezależnie od jego typu
    host_length = len(hostname)

    if is_ip:
        # Adres IP nie ma domeny, TLD ani subdomeny
        domain = ""
        tld = ""
        suffix = ""
        subdomain = ""

    else:
        # Standardowa ekstrakcja domeny
        extracted = extractor(hostname)
        domain = extracted.domain or ""
        tld = (
            extracted.suffix.rsplit(".", 1)[-1]
            if extracted.suffix else ""
        )
        suffix=extracted.suffix
        subdomain = extracted.subdomain or ""

    # Długości
    url_length = len(url)
    domain_length = len(domain)
    subdomain_length = len(subdomain)
    path_length = len(path)
    arguments_length = len(query)
    suffix_length = len(suffix)

    url_tokens = [
        token
        for token in TOKEN_SEPARATOR.split(url)
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
        "domain": domain,
        "tld": tld,
        "suffix": suffix,
        "subdomain": subdomain,
        "url_length": url_length,
        "domain_length": domain_length,
        "subdomain_length": subdomain_length,
        "suffix_length": suffix_length,
        "host_length": host_length,
        "path_length": path_length,
        "arguments_url_ratio": safe_ratio(
            arguments_length, url_length
        ),
        "path_url_ratio": safe_ratio(
            path_length, url_length
        ),
        "arguments_domain_ratio": safe_ratio(
            arguments_length, domain_length
        ),
        "arguments_hostname_ratio": safe_ratio(
            arguments_length, host_length
        ),
        "subdomain_domain_ratio": safe_ratio(
            subdomain_length, domain_length
        ),
        "suffix_tld_ratio": safe_ratio(
            suffix_length, len(tld)
        ),
        "longest_url_token_length": longest_token_length,
        "average_url_token_length": average_token_length,
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "underscore_count": url.count("_"),
        "question_mark_count": url.count("?"),
        "equal_sign_count": url.count("="),
        "ampersand_count": url.count("&"),
        "has_https": int(
            url.startswith("https://")
        ),
        "domain_entropy": shannon_entropy(domain),
        "subdomain_entropy": shannon_entropy(subdomain),
        "symbol_count": sum(
            not char.isalnum() for char in url
        ),
        "digit_count": sum(
            char.isdigit() for char in url
        ),
        "is_ip": is_ip
    }

    return features


# ========== PRZETWARZANIE DATASETU ==========

def main():
    # 1. Wczytanie danych
    df = pd.read_csv(INPUT_FILE, keep_default_na=False)
    print(f"Wczytano {len(df)} rekordów.")

    # 2. Ekstrakcja
    features_df = pd.DataFrame(
        df[URL_COLUMN].apply(extract_features).tolist(),
        index=df.index
    )
    # 3. Dodanie cech do oryginalnego DataFrame
    df = pd.concat([df, features_df], axis=1)
    # 4. Dodatkowe mapowanie etykiety na wartość liczbową
    df['label'] = df['type'].map(label_mapping)
    # 5. Zapis
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # 6. Podsumowanie
    print(f"Wyekstraktowano {features_df.shape[1]} cech.")
    print(f"Zapisano dane do: {OUTPUT_FILE}")
    print("\nBrakujące wartości w nowych cechach:")
    print(features_df.isna().sum())


if __name__ == "__main__":
    main()
