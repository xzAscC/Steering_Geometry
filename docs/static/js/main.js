document.addEventListener("DOMContentLoaded", () => {
  const navbarBurgers = Array.from(document.querySelectorAll(".navbar-burger"));

  navbarBurgers.forEach((burger) => {
    burger.addEventListener("click", () => {
      const targetId = burger.dataset.target;
      const target = targetId ? document.getElementById(targetId) : null;

      burger.classList.toggle("is-active");
      burger.setAttribute("aria-expanded", String(burger.classList.contains("is-active")));

      if (target) {
        target.classList.toggle("is-active");
      }
    });
  });

  const navbarMenu = document.getElementById("publicationNavbar");
  const navLinks = Array.from(document.querySelectorAll('a[href^="#"]'));

  navLinks.forEach((link) => {
    link.addEventListener("click", (event) => {
      const href = link.getAttribute("href");

      if (!href || href === "#") {
        return;
      }

      const target = document.querySelector(href);

      if (!target) {
        return;
      }

      event.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      history.pushState(null, "", href);

      navbarBurgers.forEach((burger) => {
        burger.classList.remove("is-active");
        burger.setAttribute("aria-expanded", "false");
      });

      if (navbarMenu) {
        navbarMenu.classList.remove("is-active");
      }
    });
  });

  const copyButton = document.getElementById("copy-bibtex");
  const bibtexCode = document.getElementById("bibtex-code");

  if (copyButton && bibtexCode) {
    const copyLabel = copyButton.querySelector(".copy-label");
    const originalText = copyLabel ? copyLabel.textContent : "Copy";

    copyButton.addEventListener("click", async () => {
      const citation = bibtexCode.textContent || "";
      const copied = await copyText(citation);

      if (copyLabel) {
        copyLabel.textContent = copied ? "Copied!" : "Copy failed";
      }

      copyButton.classList.toggle("is-success", copied);
      copyButton.classList.toggle("is-danger", !copied);

      window.setTimeout(() => {
        if (copyLabel) {
          copyLabel.textContent = originalText;
        }
        copyButton.classList.remove("is-success", "is-danger");
      }, 2000);
    });
  }

  const sectionLinks = Array.from(document.querySelectorAll('.navbar-end a[href^="#"]'));
  const sections = sectionLinks
    .map((link) => {
      const href = link.getAttribute("href");
      return href ? document.querySelector(href) : null;
    })
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length > 0) {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) {
            return;
          }

          const activeId = `#${entry.target.id}`;
          sectionLinks.forEach((link) => {
            link.classList.toggle("is-active", link.getAttribute("href") === activeId);
          });
        });
      },
      { rootMargin: "-35% 0px -55% 0px", threshold: 0.01 },
    );

    sections.forEach((section) => observer.observe(section));
  }
});

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (error) {
      return fallbackCopy(text);
    }
  }

  return fallbackCopy(text);
}

function fallbackCopy(text) {
  const textArea = document.createElement("textarea");
  textArea.value = text;
  textArea.setAttribute("readonly", "");
  textArea.style.position = "fixed";
  textArea.style.top = "-9999px";
  textArea.style.left = "-9999px";
  document.body.appendChild(textArea);
  textArea.select();

  let copied = false;

  try {
    copied = document.execCommand("copy");
  } catch (error) {
    copied = false;
  }

  document.body.removeChild(textArea);
  return copied;
}
