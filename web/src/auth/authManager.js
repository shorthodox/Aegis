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

    static getSubscriptionStatus() {
        const user = this.getUser();
        if (!user) return 'expired';
        
        // If pro, they have full access
        if (user.plan === 'pro') return 'active';
        
        // If trial, check if it's expired
        if (user.trial && user.trial.endDate) {
            const endDate = new Date(user.trial.endDate);
            if (new Date() < endDate) {
                return 'active';
            }
        }
        
        return 'expired';
    }
}

