export class AuthManager {
    static getToken() {
        return localStorage.getItem('access_token') || sessionStorage.getItem('access_token');
    }

    static setToken(token, remember = true) {
        if (remember) {
            localStorage.setItem('access_token', token);
        } else {
            sessionStorage.setItem('access_token', token);
        }
    }
    
    static clearToken() {
        localStorage.removeItem('access_token');
        sessionStorage.removeItem('access_token');
    }
    
    static isAuthenticated() {
        return !!this.getToken();
    }
    
    static getAuthHeader() {
        const token = this.getToken();
        return token ? `Bearer ${token}` : null;
    }

    static setUser(userData) {
        localStorage.setItem('user_profile', JSON.stringify(userData));
        if (userData && userData.trial && userData.trial.endDate) {
            // Support both Firestore Timestamp and ISO String
            let endDate = userData.trial.endDate;
            if (typeof endDate === 'object' && endDate.seconds) {
                endDate = new Date(endDate.seconds * 1000).toISOString();
            }
            localStorage.setItem('trial_end_timestamp', endDate);
        }
    }

    static getUser() {
        const data = localStorage.getItem('user_profile');
        return data ? JSON.parse(data) : null;
    }

    static getPlanType() {
        const user = this.getUser();
        if (!user) return 'free_trial';
        return user.plan || 'free_trial';
    }

    static isTrialValid() {
        const trialEnd = localStorage.getItem('trial_end_timestamp');
        if (!trialEnd) return false;
        return new Date() < new Date(trialEnd);
    }

    static isTokenValid() {
        const token = this.getToken();
        if (!token) return false;
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            // Add a 5 minute buffer for token expiration
            return payload.exp * 1000 > Date.now() + 300000;
        } catch (e) {
            return false;
        }
    }

    static async refreshToken() {
        try {
            const response = await fetch('/api/auth/refresh', {
                method: 'POST',
                headers: {
                    'Authorization': this.getAuthHeader()
                }
            });
            if (response.ok) {
                const data = await response.json();
                if (data.access_token) {
                    this.setToken(data.access_token);
                    return true;
                }
            }
            return false;
        } catch (error) {
            console.error('Token refresh failed', error);
            return false;
        }
    }

    static getSubscriptionStatus() {
        const user = this.getUser();
        if (!user) return 'expired';
        
        // If pro, they have full access
        if (user.plan === 'pro') return 'active';
        
        // If trial, check if it's valid
        if (this.isTrialValid()) {
            return 'active';
        }
        
        return 'expired';
    }
}
